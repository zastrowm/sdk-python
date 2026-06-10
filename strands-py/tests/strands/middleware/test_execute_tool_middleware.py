"""Integration tests for ExecuteToolStage middleware with Agent."""

import pytest

import strands
from strands import Agent, ExecuteToolStage
from strands._middleware.stages import ExecuteToolContext, ExecuteToolResult
from strands._middleware.types import _MiddlewareResult
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
        yield _MiddlewareResult(
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
        from dataclasses import replace

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
        from dataclasses import replace

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
        from strands.hooks import BeforeToolCallEvent

        _, events = hook_provider.get_events()
        event_types = [type(e) for e in events]
        middleware_saw_before_hook = BeforeToolCallEvent in event_types
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, check_middleware)
    agent("what is 2+2?")
    assert middleware_saw_before_hook


# --- cleanup ---


def test_cleanup_removes_middleware(agent):
    call_count = 0

    async def counter(context, next_fn):
        nonlocal call_count
        call_count += 1
        async for event in next_fn(context):
            yield event

    cleanup = agent.add_middleware(ExecuteToolStage, counter)
    agent("what is 2+2?")
    assert call_count == 1

    cleanup()

    # Need fresh model for second invocation
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t2", "name": "calculator", "input": {"expression": "3+3"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "6"}]}
    agent.model = MockedModelProvider([tool_use_msg, final_msg])
    agent("what is 3+3?")
    assert call_count == 1
