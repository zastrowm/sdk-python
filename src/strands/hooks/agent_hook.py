from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Protocol, TYPE_CHECKING, List, Type, Dict, Any, TypeVar, ParamSpec, Callable, Generic, \
    overload, Iterable

from strands.types.tools import AgentTool


if TYPE_CHECKING:
    from strands import Agent


class Reference(Generic[T]):
    def __init__(self, value: T) -> None:
        self.value = value

class AgentHook(Protocol):
    """Protocol for hooks that can be registered with an AgentHookManager.

    Hooks provide a way to extend agent functionality by subscribing to various events
    in the agent lifecycle.
    """
    def register_hooks(self, hooks: "AgentHookManager", agent: "Agent") -> None: ...

T = TypeVar('T', bound=Callable)


@dataclass
class HookEvent(Protocol):
    """Base protocol for all hook events.

    All hook events must include a reference to the agent that triggered them.
    """
    agent: "Agent"

@dataclass
class AgentInitializedHookEvent(HookEvent):
    ...

@dataclass
class TextDiffsHookEvent(HookEvent):
    data: str

@dataclass
class ToolCallHookEvent(HookEvent):
    tool: AgentTool
    tool_input: str
    tool_response: str
    tool_response_metadata: dict[str, Any]

TEvent = TypeVar('TEvent', bound=HookEvent)


class HookCallback(Protocol, Generic(TEvent)):
    """Protocol for callback functions that handle hook events.

    Callbacks are called when a specific event type is triggered.
    """
    def __call__(self, event: TEvent) -> None: ...

OrderedHookCallback = namedtuple('OrderedHookCallback', ['callback', 'order'])

# Consider AgentContext (per agent), RequestContext (per turn), ToolContext (per tool)


class AgentHookManager:
    """Manages event-based hooks for agents.

    The AgentHookManager provides a flexible system for registering and triggering event-based
    callbacks within the agent lifecycle. 

    Hooks are organized by event types, allowing different components to subscribe to specific
    events and respond accordingly. The manager ensures that callbacks are executed in the
    correct order and propagates events through the hierarchy.
    """

    _registered_hooks: Dict[Type, List[OrderedHookCallback]]

    def __init__(self, agent: "Agent", parent: Optional["AgentHookManager"] = None, hooks: Optional[List[AgentHook]] = None) -> None:
        """Initialize a new hook manager for an agent.

        Args:
            agent: The agent this hook manager belongs to.
            parent: Optional parent hook manager for hierarchical hook management.
            hooks: Optional list of hooks to register immediately.
        """
        self._agent = agent
        self._parent = parent # used when passing hooks through temporarily
        self._registered_hooks = {}

        for hook in hooks or []:
            self.register_hook(hook)

    def register_hook(self, hook: AgentHook):
        """Register a hook with this manager.

        The hook's register_hooks method will be called with this manager and the agent,
        allowing the hook to subscribe to specific events.
        """
        hook.register_hooks(hooks=self, agent=self._agent)

    def subscribe_to_event(self, event_type: Type[T], callback: HookCallback[T], order: int = 0):
        """Subscribe a callback to a specific event type.

        Registers a callback function to be called when events of the specified type are triggered.
        Callbacks can be ordered to control execution sequence.
        """
        if event_type not in self._registered_hooks:
            self._registered_hooks[event_type] = []

        self._registered_hooks[event_type].append(OrderedHookCallback(callback=callback, order=order))

    def get_hook_callbacks(self, hook_id: Type[TEvent]) -> list[Callable[[TEvent], None]]:
        """Get all callbacks for a specific event type, sorted by order.

        Collects callbacks from this manager and all parent managers.

        Args:
            hook_id: The event type to get callbacks for.

        Returns:
            A list of callback functions sorted by their execution order.
        """
        callbacks = self._add_callbacks_to_list(hook_id, [])
        sorted_callbacks = sorted(callbacks, key=lambda x: x.order)
        # Extract just the callback functions
        return [ordered_callback.callback for ordered_callback in sorted_callbacks]

    def trigger_event(self, event: TEvent) -> None:
        """Trigger an event, calling all registered callbacks.

        Executes all callbacks registered for the event's type, in order.

        Args:
            event: The event to trigger.
        """
        hook_type = type(event)

        for hook_callback in self.get_hook_callbacks(hook_type):
            hook_callback(event)

    def _add_callbacks_to_list(self, hook_id: Type[TEvent], callbacks: list[OrderedHookCallback]) -> list[
        OrderedHookCallback]:
        """Recursively collect callbacks from this manager and its parents.

        Args:
            hook_id: The event type to collect callbacks for.
            callbacks: The list to add callbacks to.

        Returns:
            The updated list of callbacks.
        """
        if self._parent is not None:
            self._parent._add_callbacks_to_list(hook_id, callbacks)

        my_callbacks = self._registered_hooks.get(hook_id)
        if my_callbacks is not None:
            callbacks.extend(my_callbacks)

        return callbacks


class TextPrinting(AgentHook):
    def register_hooks(self, hooks: "AgentHookManager", agent: "Agent") -> None:
        hooks.subscribe_to_event(TextDiffsHookEvent, self._handle_text_diffs)
        hooks.subscribe_to_event(ToolCallHookEvent, self._handle_tool_calls)

    def _handle_text_diffs(self, event: TextDiffsHookEvent) -> None:
        print(event.data)

    def _handle_tool_calls(self, event: ToolCallHookEvent) -> None:
        print(f"Tool call: {event.tool.tool_name} with input: {event.tool_input}")

###############################################################################
# Example Usage
###############################################################################

"""
The following examples demonstrate how to use the hook system in the Strands library.
These examples are for educational purposes and are not executed as part of the code.
"""

# Example 1: Creating and using a simple hook
# -------------------------------------------
# This example shows how to create a custom hook that logs agent events

class LoggingHook(AgentHook):
    """A hook that logs various agent events."""

    def register_hooks(self, hooks: "AgentHookManager", agent: "Agent") -> None:
        # Always log before other hooks in case an exception is thrown from them
        hooks.subscribe_to_event(AgentInitializedHookEvent, self._log_initialization, order = -100)
        hooks.subscribe_to_event(TextDiffsHookEvent, self._log_text, order = -100)
        hooks.subscribe_to_event(ToolCallHookEvent, self._log_tool_call, order = -100)

    def _log_initialization(self, event: AgentInitializedHookEvent) -> None:
        print(f"Agent initialized: {event.agent}")

    def _log_text(self, event: TextDiffsHookEvent) -> None:
        print(f"Text generated: {event.data[:50]}...")

    def _log_tool_call(self, event: ToolCallHookEvent) -> None:
        print(f"Tool called: {event.tool.tool_name}")
        print(f"  Input: {event.tool_input}")
        print(f"  Response: {event.tool_response[:50]}...")

# Create an instance of the hook
logger_hook = LoggingHook()

# When creating an agent, pass the hook
# agent = Agent(hooks=[logger_hook])


# Example 2: Creating a hook that stores conversation history
# ----------------------------------------------------------
# This example demonstrates a more complex hook that stores conversation history

class ConversationHistoryHook(AgentHook):
    """A hook that stores the conversation history."""

    def __init__(self):
        self.history = []

    def register_hooks(self, hooks: "AgentHookManager", agent: "Agent") -> None:
        hooks.subscribe_to_event(TextDiffsHookEvent, self._store_text)
        hooks.subscribe_to_event(ToolCallHookEvent, self._store_tool_call)

    def _store_text(self, event: TextDiffsHookEvent) -> None:
        self.history.append({"type": "text", "content": event.data})

    def _store_tool_call(self, event: ToolCallHookEvent) -> None:
        self.history.append({
            "type": "tool_call",
            "tool": event.tool.tool_name,
            "input": event.tool_input,
            "response": event.tool_response
        })

    def get_history(self):
        return self.history

# Create an instance of the hook
history_hook = ConversationHistoryHook()

# When creating an agent, pass the hook
# agent = Agent(hooks=[history_hook])

# Later, you can access the conversation history
# conversation_history = history_hook.get_history()


# Example 3: Combining multiple hooks
# ----------------------------------
# This example shows how to use multiple hooks together

# Create instances of different hooks
text_printer = TextPrinting()
logger = LoggingHook()
history_keeper = ConversationHistoryHook()

# When creating an agent, pass all hooks
# agent = Agent(hooks=[text_printer, logger, history_keeper])


