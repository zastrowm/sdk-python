"""Agent Interface.

This module implements the core Agent class that serves as the primary entry point for interacting with foundation
models and tools in the SDK.

The Agent interface supports two complementary interaction patterns:

1. Natural language for conversation: `agent("Analyze this data")`
2. Method-style for direct tool access: `agent.tool.tool_name(param1="value")`
"""

import asyncio
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Mapping, Optional, Type, TypeVar, Union, cast

from opentelemetry import trace as trace_api
from pydantic import BaseModel

from ..event_loop.event_loop import event_loop_cycle, run_tool
from ..handlers.callback_handler import PrintingCallbackHandler, null_callback_handler
from ..hooks import (
    AfterInvocationEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)
from ..models.bedrock import BedrockModel
from ..models.model import Model
from ..session.session_manager import SessionManager
from ..telemetry.metrics import EventLoopMetrics
from ..telemetry.tracer import get_tracer
from ..tools.registry import ToolRegistry
from ..tools.watcher import ToolWatcher
from ..types.content import ContentBlock, Message, Messages
from ..types.exceptions import ContextWindowOverflowException
from ..types.tools import ToolResult, ToolUse
from ..types.traces import AttributeValue
from .agent_result import AgentResult
from .conversation_manager import (
    ConversationManager,
    SlidingWindowConversationManager,
)
from .state import AgentState

logger = logging.getLogger(__name__)

# TypeVar for generic structured output
T = TypeVar("T", bound=BaseModel)


# Sentinel class and object to distinguish between explicit None and default parameter value
class _DefaultCallbackHandlerSentinel:
    """Sentinel class to distinguish between explicit None and default parameter value."""

    pass


_DEFAULT_CALLBACK_HANDLER = _DefaultCallbackHandlerSentinel()
_DEFAULT_AGENT_NAME = "Strands Agents"
_DEFAULT_AGENT_ID = "default"


class Agent:
    """Core Agent interface.

    An agent orchestrates the following workflow:

    1. Receives user input
    2. Processes the input using a language model
    3. Decides whether to use tools to gather information or perform actions
    4. Executes those tools and receives results
    5. Continues reasoning with the new information
    6. Produces a final response
    """

    class ToolCaller:
        """Call tool as a function."""

        def __init__(self, agent: "Agent") -> None:
            """Initialize instance.

            Args:
                agent: Agent reference that will accept tool results.
            """
            # WARNING: Do not add any other member variables or methods as this could result in a name conflict with
            #          agent tools and thus break their execution.
            self._agent = agent

        def __getattr__(self, name: str) -> Callable[..., Any]:
            """Call tool as a function.

            This method enables the method-style interface (e.g., `agent.tool.tool_name(param="value")`).
            It matches underscore-separated names to hyphenated tool names (e.g., 'some_thing' matches 'some-thing').

            Args:
                name: The name of the attribute (tool) being accessed.

            Returns:
                A function that when called will execute the named tool.

            Raises:
                AttributeError: If no tool with the given name exists or if multiple tools match the given name.
            """

            def caller(
                user_message_override: Optional[str] = None,
                record_direct_tool_call: Optional[bool] = None,
                **kwargs: Any,
            ) -> Any:
                """Call a tool directly by name.

                Args:
                    user_message_override: Optional custom message to record instead of default
                    record_direct_tool_call: Whether to record direct tool calls in message history. Overrides class
                        attribute if provided.
                    **kwargs: Keyword arguments to pass to the tool.

                Returns:
                    The result returned by the tool.

                Raises:
                    AttributeError: If the tool doesn't exist.
                """
                normalized_name = self._find_normalized_tool_name(name)

                # Create unique tool ID and set up the tool request
                tool_id = f"tooluse_{name}_{random.randint(100000000, 999999999)}"
                tool_use: ToolUse = {
                    "toolUseId": tool_id,
                    "name": normalized_name,
                    "input": kwargs.copy(),
                }

                async def acall() -> ToolResult:
                    # Pass kwargs as invocation_state
                    async for event in run_tool(self._agent, tool_use, kwargs):
                        _ = event

                    return cast(ToolResult, event)

                def tcall() -> ToolResult:
                    return asyncio.run(acall())

                with ThreadPoolExecutor() as executor:
                    future = executor.submit(tcall)
                    tool_result = future.result()

                if record_direct_tool_call is not None:
                    should_record_direct_tool_call = record_direct_tool_call
                else:
                    should_record_direct_tool_call = self._agent.record_direct_tool_call

                if should_record_direct_tool_call:
                    # Create a record of this tool execution in the message history
                    self._agent._record_tool_execution(tool_use, tool_result, user_message_override)

                # Apply window management
                self._agent.conversation_manager.apply_management(self._agent)

                return tool_result

            return caller

        def _find_normalized_tool_name(self, name: str) -> str:
            """Lookup the tool represented by name, replacing characters with underscores as necessary."""
            tool_registry = self._agent.tool_registry.registry

            if tool_registry.get(name, None):
                return name

            # If the desired name contains underscores, it might be a placeholder for characters that can't be
            # represented as python identifiers but are valid as tool names, such as dashes. In that case, find
            # all tools that can be represented with the normalized name
            if "_" in name:
                filtered_tools = [
                    tool_name for (tool_name, tool) in tool_registry.items() if tool_name.replace("-", "_") == name
                ]

                # The registry itself defends against similar names, so we can just take the first match
                if filtered_tools:
                    return filtered_tools[0]

            raise AttributeError(f"Tool '{name}' not found")

    def __init__(
        self,
        model: Union[Model, str, None] = None,
        messages: Optional[Messages] = None,
        tools: Optional[list[Union[str, dict[str, str], Any]]] = None,
        system_prompt: Optional[str] = None,
        callback_handler: Optional[
            Union[Callable[..., Any], _DefaultCallbackHandlerSentinel]
        ] = _DEFAULT_CALLBACK_HANDLER,
        conversation_manager: Optional[ConversationManager] = None,
        record_direct_tool_call: bool = True,
        load_tools_from_directory: bool = False,
        trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        *,
        agent_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        state: Optional[Union[AgentState, dict]] = None,
        hooks: Optional[list[HookProvider]] = None,
        session_manager: Optional[SessionManager] = None,
    ):
        """Initialize the Agent with the specified configuration.

        Args:
            model: Provider for running inference or a string representing the model-id for Bedrock to use.
                Defaults to strands.models.BedrockModel if None.
            messages: List of initial messages to pre-load into the conversation.
                Defaults to an empty list if None.
            tools: List of tools to make available to the agent.
                Can be specified as:

                - String tool names (e.g., "retrieve")
                - File paths (e.g., "/path/to/tool.py")
                - Imported Python modules (e.g., from strands_tools import current_time)
                - Dictionaries with name/path keys (e.g., {"name": "tool_name", "path": "/path/to/tool.py"})
                - Functions decorated with `@strands.tool` decorator.

                If provided, only these tools will be available. If None, all tools will be available.
            system_prompt: System prompt to guide model behavior.
                If None, the model will behave according to its default settings.
            callback_handler: Callback for processing events as they happen during agent execution.
                If not provided (using the default), a new PrintingCallbackHandler instance is created.
                If explicitly set to None, null_callback_handler is used.
            conversation_manager: Manager for conversation history and context window.
                Defaults to strands.agent.conversation_manager.SlidingWindowConversationManager if None.
            record_direct_tool_call: Whether to record direct tool calls in message history.
                Defaults to True.
            load_tools_from_directory: Whether to load and automatically reload tools in the `./tools/` directory.
                Defaults to False.
            trace_attributes: Custom trace attributes to apply to the agent's trace span.
            agent_id: Optional ID for the agent, useful for session management and multi-agent scenarios.
                Defaults to "default".
            name: name of the Agent
                Defaults to "Strands Agents".
            description: description of what the Agent does
                Defaults to None.
            state: stateful information for the agent. Can be either an AgentState object, or a json serializable dict.
                Defaults to an empty AgentState object.
            hooks: hooks to be added to the agent hook registry
                Defaults to None.
            session_manager: Manager for handling agent sessions including conversation history and state.
                If provided, enables session-based persistence and state management.
        """
        self.model = BedrockModel() if not model else BedrockModel(model_id=model) if isinstance(model, str) else model
        self.messages = messages if messages is not None else []

        self.system_prompt = system_prompt
        self.agent_id = agent_id or _DEFAULT_AGENT_ID
        self.name = name or _DEFAULT_AGENT_NAME
        self.description = description

        # If not provided, create a new PrintingCallbackHandler instance
        # If explicitly set to None, use null_callback_handler
        # Otherwise use the passed callback_handler
        self.callback_handler: Union[Callable[..., Any], PrintingCallbackHandler]
        if isinstance(callback_handler, _DefaultCallbackHandlerSentinel):
            self.callback_handler = PrintingCallbackHandler()
        elif callback_handler is None:
            self.callback_handler = null_callback_handler
        else:
            self.callback_handler = callback_handler

        self.conversation_manager = conversation_manager if conversation_manager else SlidingWindowConversationManager()

        # Process trace attributes to ensure they're of compatible types
        self.trace_attributes: dict[str, AttributeValue] = {}
        if trace_attributes:
            for k, v in trace_attributes.items():
                if isinstance(v, (str, int, float, bool)) or (
                    isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v)
                ):
                    self.trace_attributes[k] = v

        self.record_direct_tool_call = record_direct_tool_call
        self.load_tools_from_directory = load_tools_from_directory

        self.tool_registry = ToolRegistry()

        # Process tool list if provided
        if tools is not None:
            self.tool_registry.process_tools(tools)

        # Initialize tools and configuration
        self.tool_registry.initialize_tools(self.load_tools_from_directory)
        if load_tools_from_directory:
            self.tool_watcher = ToolWatcher(tool_registry=self.tool_registry)

        self.event_loop_metrics = EventLoopMetrics()

        # Initialize tracer instance (no-op if not configured)
        self.tracer = get_tracer()
        self.trace_span: Optional[trace_api.Span] = None

        # Initialize agent state management
        if state is not None:
            if isinstance(state, dict):
                self.state = AgentState(state)
            elif isinstance(state, AgentState):
                self.state = state
            else:
                raise ValueError("state must be an AgentState object or a dict")
        else:
            self.state = AgentState()

        self.tool_caller = Agent.ToolCaller(self)

        self.hooks = HookRegistry()

        # Initialize session management functionality
        self._session_manager = session_manager
        if self._session_manager:
            self.hooks.add_hook(self._session_manager)

        if hooks:
            for hook in hooks:
                self.hooks.add_hook(hook)
        self.hooks.invoke_callbacks(AgentInitializedEvent(agent=self))

    @property
    def tool(self) -> ToolCaller:
        """Call tool as a function.

        Returns:
            Tool caller through which user can invoke tool as a function.

        Example:
            ```
            agent = Agent(tools=[calculator])
            agent.tool.calculator(...)
            ```
        """
        return self.tool_caller

    @property
    def tool_names(self) -> list[str]:
        """Get a list of all registered tool names.

        Returns:
            Names of all tools available to this agent.
        """
        all_tools = self.tool_registry.get_all_tools_config()
        return list(all_tools.keys())

    def __call__(self, prompt: Union[str, list[ContentBlock]], **kwargs: Any) -> AgentResult:
        """Process a natural language prompt through the agent's event loop.

        This method implements the conversational interface (e.g., `agent("hello!")`). It adds the user's prompt to
        the conversation history, processes it through the model, executes any tool calls, and returns the final result.

        Args:
            prompt: User input as text or list of ContentBlock objects for multi-modal content.
            **kwargs: Additional parameters to pass through the event loop.

        Returns:
            Result object containing:

                - stop_reason: Why the event loop stopped (e.g., "end_turn", "max_tokens")
                - message: The final message from the model
                - metrics: Performance metrics from the event loop
                - state: The final state of the event loop

        Raises:
            ValueError: If prompt is None.
        """

        def execute() -> AgentResult:
            return asyncio.run(self.invoke_async(prompt, **kwargs))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    async def invoke_async(self, prompt: Union[str, list[ContentBlock]], **kwargs: Any) -> AgentResult:
        """Process a natural language prompt through the agent's event loop.

        This method implements the conversational interface (e.g., `agent("hello!")`). It adds the user's prompt to
        the conversation history, processes it through the model, executes any tool calls, and returns the final result.

        Args:
            prompt: User input as text or list of ContentBlock objects for multi-modal content.
            **kwargs: Additional parameters to pass through the event loop.

        Returns:
            Result object containing:

                - stop_reason: Why the event loop stopped (e.g., "end_turn", "max_tokens")
                - message: The final message from the model
                - metrics: Performance metrics from the event loop
                - state: The final state of the event loop

        Raises:
            ValueError: If prompt is None.
        """
        events = self.stream_async(prompt, **kwargs)
        async for event in events:
            _ = event

        return cast(AgentResult, event["result"])

    def structured_output(self, output_model: Type[T], prompt: Optional[Union[str, list[ContentBlock]]] = None) -> T:
        """This method allows you to get structured output from the agent.

        If you pass in a prompt, it will be used temporarily without adding it to the conversation history.
        If you don't pass in a prompt, it will use only the existing conversation history to respond.

        For smaller models, you may want to use the optional prompt to add additional instructions to explicitly
        instruct the model to output the structured data.

        Args:
            output_model: The output model (a JSON schema written as a Pydantic BaseModel)
                that the agent will use when responding.
            prompt: The prompt to use for the agent (will not be added to conversation history).

        Raises:
            ValueError: If no conversation history or prompt is provided.
        """

        def execute() -> T:
            return asyncio.run(self.structured_output_async(output_model, prompt))

        with ThreadPoolExecutor() as executor:
            future = executor.submit(execute)
            return future.result()

    async def structured_output_async(
        self, output_model: Type[T], prompt: Optional[Union[str, list[ContentBlock]]] = None
    ) -> T:
        """This method allows you to get structured output from the agent.

        If you pass in a prompt, it will be used temporarily without adding it to the conversation history.
        If you don't pass in a prompt, it will use only the existing conversation history to respond.

        For smaller models, you may want to use the optional prompt to add additional instructions to explicitly
        instruct the model to output the structured data.

        Args:
            output_model: The output model (a JSON schema written as a Pydantic BaseModel)
                that the agent will use when responding.
            prompt: The prompt to use for the agent (will not be added to conversation history).

        Raises:
            ValueError: If no conversation history or prompt is provided.
        """
        self.hooks.invoke_callbacks(BeforeInvocationEvent(agent=self))

        try:
            if not self.messages and not prompt:
                raise ValueError("No conversation history or prompt provided")

            # Create temporary messages array if prompt is provided
            if prompt:
                temp_messages = self.messages + [self._standardize_prompt(prompt)]
            else:
                temp_messages = self.messages

            events = self.model.structured_output(output_model, temp_messages, system_prompt=self.system_prompt)
            async for event in events:
                if "callback" in event:
                    self.callback_handler(**cast(dict, event["callback"]))

            return event["output"]

        finally:
            self.hooks.invoke_callbacks(AfterInvocationEvent(agent=self))

    async def stream_async(self, prompt: Union[str, list[ContentBlock]], **kwargs: Any) -> AsyncIterator[Any]:
        """Process a natural language prompt and yield events as an async iterator.

        This method provides an asynchronous interface for streaming agent events, allowing
        consumers to process stream events programmatically through an async iterator pattern
        rather than callback functions. This is particularly useful for web servers and other
        async environments.

        Args:
            prompt: User input as text or list of ContentBlock objects for multi-modal content.
            **kwargs: Additional parameters to pass to the event loop.

        Yields:
            An async iterator that yields events. Each event is a dictionary containing
            information about the current state of processing, such as:

            - data: Text content being generated
            - complete: Whether this is the final chunk
            - current_tool_use: Information about tools being executed
            - And other event data provided by the callback handler

        Raises:
            ValueError: If prompt is None.
            Exception: Any exceptions from the agent invocation will be propagated to the caller.

        Example:
            ```python
            async for event in agent.stream_async("Analyze this data"):
                if "data" in event:
                    yield event["data"]
            ```
        """
        callback_handler = kwargs.get("callback_handler", self.callback_handler)

        message = self._standardize_prompt(prompt)

        self.trace_span = self._start_agent_trace_span(message)
        with trace_api.use_span(self.trace_span):
            try:
                events = self._run_loop(message, invocation_state=kwargs)
                async for event in events:
                    if "callback" in event:
                        callback_handler(**event["callback"])
                        yield event["callback"]

                result = AgentResult(*event["stop"])
                callback_handler(result=result)
                yield {"result": result}

                self._end_agent_trace_span(response=result)

            except Exception as e:
                self._end_agent_trace_span(error=e)
                raise

    async def _run_loop(
        self, message: Message, invocation_state: dict[str, Any]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Execute the agent's event loop with the given message and parameters.

        Args:
            message: The user message to add to the conversation.
            invocation_state: Additional parameters to pass to the event loop.

        Yields:
            Events from the event loop cycle.
        """
        self.hooks.invoke_callbacks(BeforeInvocationEvent(agent=self))

        try:
            yield {"callback": {"init_event_loop": True, **invocation_state}}

            self._append_message(message)

            # Execute the event loop cycle with retry logic for context limits
            events = self._execute_event_loop_cycle(invocation_state)
            async for event in events:
                # Signal from the model provider that the message sent by the user should be redacted,
                # likely due to a guardrail.
                if (
                    event.get("callback")
                    and event["callback"].get("event")
                    and event["callback"]["event"].get("redactContent")
                    and event["callback"]["event"]["redactContent"].get("redactUserContentMessage")
                ):
                    self.messages[-1]["content"] = [
                        {"text": event["callback"]["event"]["redactContent"]["redactUserContentMessage"]}
                    ]
                    if self._session_manager:
                        self._session_manager.redact_latest_message(self.messages[-1], self)
                yield event

        finally:
            self.conversation_manager.apply_management(self)
            self.hooks.invoke_callbacks(AfterInvocationEvent(agent=self))

    def _standardize_prompt(self, prompt: Union[str, list[ContentBlock]]) -> Message:
        """Convert the prompt into a Message, validating it along the way."""
        if prompt is None:
            raise ValueError("User prompt must not be None")

        content: list[ContentBlock] = [{"text": prompt}] if isinstance(prompt, str) else prompt
        message: Message = {"role": "user", "content": content}
        return message

    async def _execute_event_loop_cycle(self, invocation_state: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
        """Execute the event loop cycle with retry logic for context window limits.

        This internal method handles the execution of the event loop cycle and implements
        retry logic for handling context window overflow exceptions by reducing the
        conversation context and retrying.

        Yields:
            Events of the loop cycle.
        """
        # Add `Agent` to invocation_state to keep backwards-compatibility
        invocation_state["agent"] = self

        try:
            # Execute the main event loop cycle
            events = event_loop_cycle(
                agent=self,
                invocation_state=invocation_state,
            )
            async for event in events:
                yield event

        except ContextWindowOverflowException as e:
            # Try reducing the context size and retrying
            self.conversation_manager.reduce_context(self, e=e)

            # Sync agent after reduce_context to keep conversation_manager_state up to date in the session
            if self._session_manager:
                self._session_manager.sync_agent(self)

            events = self._execute_event_loop_cycle(invocation_state)
            async for event in events:
                yield event

    def _record_tool_execution(
        self,
        tool: ToolUse,
        tool_result: ToolResult,
        user_message_override: Optional[str],
    ) -> None:
        """Record a tool execution in the message history.

        Creates a sequence of messages that represent the tool execution:

        1. A user message describing the tool call
        2. An assistant message with the tool use
        3. A user message with the tool result
        4. An assistant message acknowledging the tool call

        Args:
            tool: The tool call information.
            tool_result: The result returned by the tool.
            user_message_override: Optional custom message to include.
        """
        # Create user message describing the tool call
        input_parameters = json.dumps(tool["input"], default=lambda o: f"<<non-serializable: {type(o).__qualname__}>>")

        user_msg_content: list[ContentBlock] = [
            {"text": (f"agent.tool.{tool['name']} direct tool call.\nInput parameters: {input_parameters}\n")}
        ]

        # Add override message if provided
        if user_message_override:
            user_msg_content.insert(0, {"text": f"{user_message_override}\n"})

        # Create the message sequence
        user_msg: Message = {
            "role": "user",
            "content": user_msg_content,
        }
        tool_use_msg: Message = {
            "role": "assistant",
            "content": [{"toolUse": tool}],
        }
        tool_result_msg: Message = {
            "role": "user",
            "content": [{"toolResult": tool_result}],
        }
        assistant_msg: Message = {
            "role": "assistant",
            "content": [{"text": f"agent.tool.{tool['name']} was called."}],
        }

        # Add to message history
        self._append_message(user_msg)
        self._append_message(tool_use_msg)
        self._append_message(tool_result_msg)
        self._append_message(assistant_msg)

    def _start_agent_trace_span(self, message: Message) -> trace_api.Span:
        """Starts a trace span for the agent.

        Args:
            message: The user message.
        """
        model_id = self.model.config.get("model_id") if hasattr(self.model, "config") else None
        return self.tracer.start_agent_span(
            message=message,
            agent_name=self.name,
            model_id=model_id,
            tools=self.tool_names,
            system_prompt=self.system_prompt,
            custom_trace_attributes=self.trace_attributes,
        )

    def _end_agent_trace_span(
        self,
        response: Optional[AgentResult] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Ends a trace span for the agent.

        Args:
            span: The span to end.
            response: Response to record as a trace attribute.
            error: Error to record as a trace attribute.
        """
        if self.trace_span:
            trace_attributes: dict[str, Any] = {
                "span": self.trace_span,
            }

            if response:
                trace_attributes["response"] = response
            if error:
                trace_attributes["error"] = error

            self.tracer.end_agent_span(**trace_attributes)

    def _append_message(self, message: Message) -> None:
        """Appends a message to the agent's list of messages and invokes the callbacks for the MessageCreatedEvent."""
        self.messages.append(message)
        self.hooks.invoke_callbacks(MessageAddedEvent(agent=self, message=message))
