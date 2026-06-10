"""Tests for middleware-initiated interrupts on ExecuteToolStage."""

import pytest

import strands
from strands import Agent, ExecuteToolStage
from strands._middleware.stages import ExecuteToolContext, ExecuteToolResult, MiddlewareInterruptResult
from strands._middleware.types import _MiddlewareResult
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


def test_middleware_interrupt_halts_agent(agent):
    """Calling context.interrupt() raises InterruptException and halts execution."""

    async def approval_gate(context, next_fn):
        context.interrupt("approve_calc", reason="Confirm calculation?")
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, approval_gate)
    result = agent("what is 2+2?")

    assert result.stop_reason == "interrupt"
    assert len(result.interrupts) == 1
    assert result.interrupts[0].name == "approve_calc"
    assert result.interrupts[0].reason == "Confirm calculation?"


def test_middleware_interrupt_resumes_with_response(calculator_tool):
    """After providing a response, interrupt() returns the response and execution continues."""
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "tool_1", "name": "calculator", "input": {"expression": "2+2"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "The answer is 4."}]}
    model = MockedModelProvider([tool_use_msg, final_msg, final_msg])
    agent = Agent(model=model, tools=[calculator_tool], callback_handler=None)

    received_response = None

    async def approval_gate(context, next_fn):
        nonlocal received_response
        interrupt_result = context.interrupt("approve_calc", reason="Confirm?")
        received_response = interrupt_result.response
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, approval_gate)

    # First call: interrupt halts
    result = agent("what is 2+2?")
    assert result.stop_reason == "interrupt"

    # Resume with response
    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "yes"}}])
    assert received_response == "yes"


def test_middleware_interrupt_returns_middleware_interrupt_result(calculator_tool):
    """interrupt() returns a MiddlewareInterruptResult instance."""
    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "tool_1", "name": "calculator", "input": {"expression": "2+2"}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "4"}]}
    model = MockedModelProvider([tool_use_msg, final_msg, final_msg])
    agent = Agent(model=model, tools=[calculator_tool], callback_handler=None)

    interrupt_result_type = None

    async def gate(context, next_fn):
        nonlocal interrupt_result_type
        r = context.interrupt("gate", reason="check")
        interrupt_result_type = type(r)
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, gate)

    # Halt
    result = agent("calc")
    assert result.stop_reason == "interrupt"

    # Resume
    result = agent([{"interruptResponse": {"interruptId": result.interrupts[0].id, "response": "ok"}}])
    assert interrupt_result_type is MiddlewareInterruptResult


def test_middleware_interrupt_with_preemptive_response(agent):
    """Providing a preemptive response skips the interrupt entirely."""
    skipped_interrupt = False

    async def gate_with_default(context, next_fn):
        nonlocal skipped_interrupt
        r = context.interrupt("gate", reason="check", response="pre-approved")
        skipped_interrupt = r.response == "pre-approved"
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, gate_with_default)
    result = agent("what is 2+2?")

    # Should NOT interrupt — preemptive response bypasses it
    assert result.stop_reason == "end_turn"
    assert skipped_interrupt


def test_middleware_interrupt_short_circuits_tool_execution(agent):
    """When middleware interrupts, the tool does NOT execute."""
    tool_executed = False

    @strands.tool(name="tracked_tool")
    def tracked_tool() -> str:
        """A tool that tracks execution."""
        nonlocal tool_executed
        tool_executed = True
        return "done"

    tool_use_msg = {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "tracked_tool", "input": {}}}],
    }
    final_msg = {"role": "assistant", "content": [{"text": "ok"}]}
    model = MockedModelProvider([tool_use_msg, final_msg])
    agent = Agent(model=model, tools=[tracked_tool], callback_handler=None)

    async def blocker(context, next_fn):
        context.interrupt("block", reason="nope")
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, blocker)
    result = agent("do it")

    assert result.stop_reason == "interrupt"
    assert not tool_executed


def test_middleware_interrupt_id_is_deterministic(agent):
    """Same name produces same interrupt ID across invocations."""
    interrupt_ids: list[str] = []

    async def gate(context, next_fn):
        context.interrupt("my_gate", reason="check")
        async for event in next_fn(context):
            yield event

    agent.add_middleware(ExecuteToolStage, gate)

    result = agent("what is 2+2?")
    interrupt_ids.append(result.interrupts[0].id)

    # Re-invoke with same tool_use_id → same interrupt ID
    assert "middleware:executeTool:tool_1:" in interrupt_ids[0]
