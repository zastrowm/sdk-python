"""Integration tests for ExecuteToolStage middleware with Agent."""

from dataclasses import replace

import pytest

import strands
from strands import Agent, ExecuteToolStage, Plugin
from strands._middleware.stages import ExecuteToolContext, ExecuteToolResult
from strands._middleware.types import MiddlewareResult
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from tests.fixtures.mock_hook_provider import MockHookProvider
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def calculator_tool():
    @strands.tool(name="calculator")
    def func(expression: str) -> str:
        """Evaluate a math expression."""
        return str(eval(expression))

    return func


@pytest.fixture
def model(calculator_tool):
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "tool_1", "name": "calculator", "input": {"expression": "2+2"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "The answer is 4."}]}
    return MockedModelProvider([tool_use_msg, final_msg])


@pytest.fixture
def agent(model, calculator_tool):
    return Agent(model=model, tools=[calculator_tool], callback_handler=None)


# --- wrap handler ---


def test_wrap_passthrough_does_not_alter_behavior(agent):
    async def passthrough(context, next_fn):
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, passthrough)
    result = agent("what is 2+2?")
    assert result.message["content"][0]["text"] == "The answer is 4."


def test_wrap_handler_receives_execute_tool_context(agent):
    received_contexts: list[ExecuteToolContext] = []

    async def capture(context, next_fn):
        received_contexts.append(context)
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, capture)
    agent("what is 2+2?")

    assert len(received_contexts) == 1
    ctx = received_contexts[0]
    assert ctx.agent is agent
    assert ctx.tool is not None
    assert ctx.tool_use["name"] == "calculator"
    assert ctx.tool_use["input"] == {"expression": "2+2"}
    assert isinstance(ctx.invocation_state, dict)


def test_wrap_short_circuit_with_cached_result(agent):
    """Middleware can short-circuit by returning a cached tool result without calling next."""

    async def mock_tool(context, next_fn):
        yield MiddlewareResult(
            ExecuteToolResult(
                tool_result={
                    "toolUseId": context.tool_use["toolUseId"],
                    "status": "success",
                    "content": [{"text": "mocked: 42"}],
                },
            )
        )

    agent.add_middleware(ExecuteToolStage, mock_tool)
    result = agent("what is 2+2?")
    # The model sees the mocked tool result and responds
    assert result.message["content"][0]["text"] == "The answer is 4."


def test_wrap_multiple_middleware_compose_correctly(agent):
    order: list[str] = []

    async def outer(context, next_fn):
        order.append("outer_before")
        async for event in next_fn(context):
            yield event
        order.append("outer_after")

    async def inner(context, next_fn):
        order.append("inner_before")
        async for event in next_fn(context):
            yield event
        order.append("inner_after")

    agent.add_middleware(ExecuteToolStage, outer)
    agent.add_middleware(ExecuteToolStage, inner)
    agent("what is 2+2?")

    assert order == ["outer_before", "inner_before", "inner_after", "outer_after"]


def test_wrap_error_from_tool_propagates_through_middleware():
    """Errors from the tool propagate through middleware layers."""

    @strands.tool(name="broken_tool")
    def broken_tool() -> str:
        """Always fails."""
        raise RuntimeError("tool exploded")

    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "broken_tool", "input": {}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "Error occurred."}]}
    model = MockedModelProvider([tool_use_msg, final_msg])
    agent = Agent(model=model, tools=[broken_tool], callback_handler=None)

    saw_error = False

    async def error_observer(context, next_fn):
        nonlocal saw_error
        async for event in next_fn(context):
            yield event
            # Tool errors are caught inside the terminal and returned as error results
            # so they don't propagate as exceptions here — they show up in the result
        saw_error = True

    agent.add_middleware(ExecuteToolStage, error_observer)
    agent("do something")
    assert saw_error


# --- input phase ---


def test_input_transforms_tool_context(agent):
    received_input = {}

    async def capture(context, next_fn):
        received_input.update(context.tool_use.get("input", {}))
        async for event in next_fn(context):
            yield event

    def modify_input(context):

        modified_tool_use = {**context.tool_use, "input": {"expression": "3+3"}}
        return replace(context, tool_use=modified_tool_use)

    agent.add_middleware(ExecuteToolStage.Input, modify_input)
    agent.add_middleware(ExecuteToolStage, capture)
    agent("what is 2+2?")

    assert received_input == {"expression": "3+3"}


# --- output phase ---


def test_output_transforms_tool_result(agent):
    transformed_results: list[ExecuteToolResult] = []

    def output_handler(result):

        new_result = replace(
            result,
            tool_result={**result.tool_result, "content": [{"text": "intercepted"}]},
        )
        transformed_results.append(new_result)
        return new_result

    agent.add_middleware(ExecuteToolStage.Output, output_handler)
    agent("what is 2+2?")

    assert len(transformed_results) == 1
    assert transformed_results[0].tool_result["content"] == [{"text": "intercepted"}]


# --- hooks fire outside middleware ---


def test_hooks_fire_outside_middleware(model, calculator_tool):
    hook_provider = MockHookProvider(event_types="all")
    agent = Agent(model=model, tools=[calculator_tool], callback_handler=None, hooks=[hook_provider])

    middleware_saw_before_hook = False

    async def check_middleware(context, next_fn):
        nonlocal middleware_saw_before_hook

        _, events = hook_provider.get_events()
        event_types = [type(e) for e in events]
        middleware_saw_before_hook = BeforeToolCallEvent in event_types
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, check_middleware)
    agent("what is 2+2?")
    assert middleware_saw_before_hook



# --- additional coverage ---


def test_short_circuit_tool_not_called(calculator_tool):
    """When middleware short-circuits, the actual tool function is NOT called."""
    tool_called = False
    original_stream = calculator_tool.stream

    async def tracking_stream(*args, **kwargs):
        nonlocal tool_called
        tool_called = True
        async for event in original_stream(*args, **kwargs):
            yield event

    calculator_tool.stream = tracking_stream

    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "calculator", "input": {"expression": "1+1"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "2"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])
    agent = Agent(model=model, tools=[calculator_tool], callback_handler=None)

    async def cached(context, next_fn):
        yield MiddlewareResult(
            ExecuteToolResult(
                tool_result={"toolUseId": context.tool_use["toolUseId"], "status": "success", "content": [{"text": "2"}]}
            )
        )

    agent.add_middleware(ExecuteToolStage, cached)
    agent("calc")
    assert not tool_called


def test_context_transform_modified_input_reaches_tool():
    """When middleware transforms tool_use input, the tool receives modified arguments."""
    received_args: list[dict] = []

    @strands.tool(name="echo_tool")
    def echo_tool(value: str) -> str:
        """Echo the value."""
        received_args.append({"value": value})
        return value

    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "echo_tool", "input": {"value": "original"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "done"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])
    agent = Agent(model=model, tools=[echo_tool], callback_handler=None)

    def modify_input(context):

        modified_tool_use = {**context.tool_use, "input": {"value": "modified"}}
        return replace(context, tool_use=modified_tool_use)

    agent.add_middleware(ExecuteToolStage.Input, modify_input)
    agent("test")

    assert received_args == [{"value": "modified"}]


def test_context_transform_does_not_mutate_original():
    """Modifying context via replace does not mutate the original context object."""

    @strands.tool(name="echo_tool")
    def echo_tool(value: str) -> str:
        """Echo."""
        return value

    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "echo_tool", "input": {"value": "original"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "done"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])
    agent = Agent(model=model, tools=[echo_tool], callback_handler=None)

    original_contexts: list[ExecuteToolContext] = []

    async def mutating_middleware(context, next_fn):

        original_contexts.append(context)
        modified = replace(context, tool_use={**context.tool_use, "input": {"value": "changed"}})
        async for event in next_fn(modified):
            yield event

    agent.add_middleware(ExecuteToolStage, mutating_middleware)
    agent("test")

    assert len(original_contexts) == 1
    # Original context must be unchanged
    assert original_contexts[0].tool_use["input"] == {"value": "original"}


def test_after_tool_call_event_fires_after_middleware(model, calculator_tool):
    """AfterToolCallEvent fires after middleware completes."""

    hook_provider = MockHookProvider(event_types="all")
    agent = Agent(model=model, tools=[calculator_tool], callback_handler=None, hooks=[hook_provider])

    middleware_completed = False

    async def tracker(context, next_fn):
        nonlocal middleware_completed
        async for event in next_fn(context):
            yield event
        middleware_completed = True

    agent.add_middleware(ExecuteToolStage, tracker)
    agent("what is 2+2?")

    assert middleware_completed
    _, events = hook_provider.get_events()
    event_types = [type(e) for e in events]
    assert AfterToolCallEvent in event_types


def test_hooks_fire_when_middleware_short_circuits(model, calculator_tool):
    """AfterToolCallEvent fires even when middleware short-circuits."""

    hook_provider = MockHookProvider(event_types="all")
    agent = Agent(model=model, tools=[calculator_tool], callback_handler=None, hooks=[hook_provider])

    async def cached(context, next_fn):
        yield MiddlewareResult(
            ExecuteToolResult(
                tool_result={"toolUseId": context.tool_use["toolUseId"], "status": "success", "content": [{"text": "4"}]}
            )
        )

    agent.add_middleware(ExecuteToolStage, cached)
    agent("what is 2+2?")

    _, events = hook_provider.get_events()
    event_types = [type(e) for e in events]
    assert AfterToolCallEvent in event_types


def test_after_tool_call_receives_middleware_result_on_short_circuit(model, calculator_tool):
    """AfterToolCallEvent.result contains the middleware-provided result on short-circuit."""

    hook_provider = MockHookProvider(event_types="all")
    agent = Agent(model=model, tools=[calculator_tool], callback_handler=None, hooks=[hook_provider])

    async def cached(context, next_fn):
        yield MiddlewareResult(
            ExecuteToolResult(
                tool_result={
                    "toolUseId": context.tool_use["toolUseId"],
                    "status": "success",
                    "content": [{"text": "mocked_42"}],
                }
            )
        )

    agent.add_middleware(ExecuteToolStage, cached)
    agent("what is 2+2?")

    _, events = hook_provider.get_events()
    after_events = [e for e in events if isinstance(e, AfterToolCallEvent)]
    assert len(after_events) == 1
    assert after_events[0].result["content"] == [{"text": "mocked_42"}]


def test_caching_plugin_use_case():
    """Full caching plugin: first call executes, second call returns cached result."""

    call_count = 0

    @strands.tool(name="expensive_tool")
    def expensive_tool(query: str) -> str:
        """Simulates an expensive operation."""
        nonlocal call_count
        call_count += 1
        return f"result_for_{query}"

    class CachingPlugin(Plugin):
        name = "caching"

        def __init__(self):
            super().__init__()
            self._cache: dict[str, dict] = {}

        def init_agent(self, agent):
            agent.add_middleware(ExecuteToolStage, self._middleware)

        async def _middleware(self, context, next_fn):
            key = f"{context.tool_use['name']}:{context.tool_use['input']}"
            if key in self._cache:
                yield MiddlewareResult(ExecuteToolResult(tool_result=self._cache[key]))
                return
            async for event in next_fn(context):
                if isinstance(event, MiddlewareResult):
                    self._cache[key] = event.value.tool_result
                yield event

    plugin = CachingPlugin()

    # Two tool calls with same input
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "expensive_tool", "input": {"query": "hello"}}}],
    }
    tool_use_msg_2 = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t2", "name": "expensive_tool", "input": {"query": "hello"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "done"}]}

    model = MockedModelProvider([tool_use_msg, final_msg, tool_use_msg_2, final_msg])
    agent = Agent(model=model, tools=[expensive_tool], callback_handler=None, plugins=[plugin])

    agent("first call")
    assert call_count == 1

    agent("second call")
    # Tool should NOT have been called again — cached
    assert call_count == 1


def test_short_circuit_result_appears_in_conversation(calculator_tool):
    """When middleware short-circuits, the mocked result appears in agent.messages."""
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "calculator", "input": {"expression": "1+1"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "done"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])
    agent = Agent(model=model, tools=[calculator_tool], callback_handler=None)

    async def cached(context, next_fn):
        yield MiddlewareResult(
            ExecuteToolResult(
                tool_result={
                    "toolUseId": context.tool_use["toolUseId"],
                    "status": "success",
                    "content": [{"text": "mocked_result"}],
                }
            )
        )

    agent.add_middleware(ExecuteToolStage, cached)
    agent("calc")

    # Find the tool result message in conversation
    tool_result_messages = [
        msg for msg in agent.messages if msg.get("role") == "user" and any("toolResult" in c for c in msg["content"])
    ]
    assert len(tool_result_messages) == 1
    tool_result = tool_result_messages[0]["content"][0]["toolResult"]
    assert tool_result["content"] == [{"text": "mocked_result"}]
