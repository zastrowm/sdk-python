"""Unit tests for MiddlewareRegistry compose and invoke mechanics."""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from strands._middleware.registry import MiddlewareRegistry
from strands._middleware.types import MiddlewareStage, _MiddlewareResult


@pytest.fixture
def registry():
    return MiddlewareRegistry()


@pytest.fixture
def stage():
    return MiddlewareStage[dict, str, str](name="test")


async def _collect(gen: AsyncGenerator[Any, None]) -> tuple[list[Any], Any]:
    """Iterate a middleware chain, separating events from the _MiddlewareResult."""
    events: list[Any] = []
    result = None
    async for item in gen:
        if isinstance(item, _MiddlewareResult):
            result = item.value
        else:
            events.append(item)
    return events, result


def _make_terminal(*events: Any, result: Any = "terminal_result"):
    """Create a terminal function that yields events then a _MiddlewareResult."""

    async def terminal(context: Any) -> AsyncGenerator[Any, None]:
        for event in events:
            yield event
        yield _MiddlewareResult(result)

    return terminal


# --- compose: no handlers ---


@pytest.mark.asyncio
async def test_compose_no_handlers_returns_terminal_directly(registry, stage):
    terminal = _make_terminal("e1", "e2", result="done")
    chain = registry.compose(stage, terminal)
    assert chain is terminal


@pytest.mark.asyncio
async def test_compose_no_handlers_events_and_result_pass_through(registry, stage):
    terminal = _make_terminal("e1", "e2", result="done")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert events == ["e1", "e2"]
    assert result == "done"


# --- wrap handler ---


@pytest.mark.asyncio
async def test_wrap_passthrough_forwards_events_and_result(registry, stage):
    async def passthrough(context, next_fn):
        async for event in next_fn(context):
            yield event

    registry.add(stage, passthrough)
    terminal = _make_terminal("e1", result="done")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert events == ["e1"]
    assert result == "done"


@pytest.mark.asyncio
async def test_wrap_context_modification_reaches_terminal(registry, stage):
    received_context = {}

    async def terminal(context):
        received_context.update(context)
        yield _MiddlewareResult("done")

    async def modifier(context, next_fn):
        async for event in next_fn({**context, "added": True}):
            yield event

    registry.add(stage, modifier)
    await _collect(registry.invoke(stage, {"original": True}, terminal))
    assert received_context == {"original": True, "added": True}


@pytest.mark.asyncio
async def test_wrap_short_circuit_skips_terminal(registry, stage):
    terminal_called = False

    async def terminal(context):
        nonlocal terminal_called
        terminal_called = True
        yield _MiddlewareResult("should not reach")

    async def short_circuit(context, next_fn):
        yield "cached_event"
        yield _MiddlewareResult("cached_result")

    registry.add(stage, short_circuit)
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert not terminal_called
    assert events == ["cached_event"]
    assert result == "cached_result"


@pytest.mark.asyncio
async def test_wrap_result_transformation(registry, stage):
    async def transformer(context, next_fn):
        async for event in next_fn(context):
            if isinstance(event, _MiddlewareResult):
                yield _MiddlewareResult(event.value.upper())
            else:
                yield event

    registry.add(stage, transformer)
    terminal = _make_terminal(result="hello")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert result == "HELLO"


@pytest.mark.asyncio
async def test_wrap_event_filtering(registry, stage):
    async def filter_middleware(context, next_fn):
        async for event in next_fn(context):
            if isinstance(event, _MiddlewareResult) or event != "skip_me":
                yield event

    registry.add(stage, filter_middleware)
    terminal = _make_terminal("keep", "skip_me", "also_keep", result="done")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert events == ["keep", "also_keep"]
    assert result == "done"


@pytest.mark.asyncio
async def test_wrap_event_injection(registry, stage):
    async def before_after_injector(context, next_fn):
        yield "before"
        result_value = None
        async for event in next_fn(context):
            if isinstance(event, _MiddlewareResult):
                result_value = event.value
            else:
                yield event
        yield "after"
        yield _MiddlewareResult(result_value)

    registry.add(stage, before_after_injector)
    terminal = _make_terminal("inner", result="done")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert events == ["before", "inner", "after"]
    assert result == "done"


@pytest.mark.asyncio
async def test_wrap_retry_calls_next_multiple_times(registry, stage):
    call_count = 0

    async def terminal(context):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("not yet")
        yield "success"
        yield _MiddlewareResult("done")

    async def retry_middleware(context, next_fn):
        for attempt in range(3):
            try:
                async for event in next_fn(context):
                    yield event
                return
            except ValueError:
                if attempt == 2:
                    raise

    registry.add(stage, retry_middleware)
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert call_count == 3
    assert events == ["success"]
    assert result == "done"


# --- composition order ---


@pytest.mark.asyncio
async def test_first_registered_is_outermost(registry, stage):
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

    registry.add(stage, outer)
    registry.add(stage, inner)
    terminal = _make_terminal(result="done")
    await _collect(registry.invoke(stage, {}, terminal))
    assert order == ["outer_before", "inner_before", "inner_after", "outer_after"]


@pytest.mark.asyncio
async def test_multiple_handlers_all_see_events(registry, stage):
    seen_by: dict[str, list] = {"a": [], "b": []}

    async def handler_a(context, next_fn):
        async for event in next_fn(context):
            if not isinstance(event, _MiddlewareResult):
                seen_by["a"].append(event)
            yield event

    async def handler_b(context, next_fn):
        async for event in next_fn(context):
            if not isinstance(event, _MiddlewareResult):
                seen_by["b"].append(event)
            yield event

    registry.add(stage, handler_a)
    registry.add(stage, handler_b)
    terminal = _make_terminal("e1", "e2", result="done")
    await _collect(registry.invoke(stage, {}, terminal))
    assert seen_by["b"] == ["e1", "e2"]
    assert seen_by["a"] == ["e1", "e2"]


# --- input phase ---


@pytest.mark.asyncio
async def test_input_transforms_context(registry, stage):
    received_context = {}

    async def terminal(context):
        received_context.update(context)
        yield _MiddlewareResult("done")

    def input_handler(context):
        return {**context, "injected": True}

    registry.add_input(stage.Input, input_handler)
    await _collect(registry.invoke(stage, {"original": True}, terminal))
    assert received_context == {"original": True, "injected": True}


@pytest.mark.asyncio
async def test_input_async_handler(registry, stage):
    received_context = {}

    async def terminal(context):
        received_context.update(context)
        yield _MiddlewareResult("done")

    async def async_input(context):
        return {**context, "async": True}

    registry.add_input(stage.Input, async_input)
    await _collect(registry.invoke(stage, {}, terminal))
    assert received_context == {"async": True}


@pytest.mark.asyncio
async def test_input_runs_before_wrap(registry, stage):
    order: list[str] = []

    async def terminal(context):
        order.append(f"terminal(injected={context.get('injected')})")
        yield _MiddlewareResult("done")

    def input_handler(context):
        order.append("input")
        return {**context, "injected": True}

    async def wrap_handler(context, next_fn):
        order.append(f"wrap(injected={context.get('injected')})")
        async for event in next_fn(context):
            yield event

    # Register wrap FIRST, then input — input still runs first due to phase ordering
    registry.add(stage, wrap_handler)
    registry.add_input(stage.Input, input_handler)
    await _collect(registry.invoke(stage, {}, terminal))
    assert order == ["input", "wrap(injected=True)", "terminal(injected=True)"]


@pytest.mark.asyncio
async def test_input_multiple_compose_in_order(registry, stage):
    received_context = {}

    async def terminal(context):
        received_context.update(context)
        yield _MiddlewareResult("done")

    def first_input(context):
        return {**context, "first": True}

    def second_input(context):
        return {**context, "second": True, "saw_first": context.get("first")}

    registry.add_input(stage.Input, first_input)
    registry.add_input(stage.Input, second_input)
    await _collect(registry.invoke(stage, {}, terminal))
    assert received_context["first"] is True
    assert received_context["second"] is True
    assert received_context["saw_first"] is True


# --- output phase ---


@pytest.mark.asyncio
async def test_output_transforms_result(registry, stage):
    def output_handler(result):
        return result + "_transformed"

    registry.add_output(stage.Output, output_handler)
    terminal = _make_terminal(result="original")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert result == "original_transformed"


@pytest.mark.asyncio
async def test_output_async_handler(registry, stage):
    async def async_output(result):
        return result + "_async"

    registry.add_output(stage.Output, async_output)
    terminal = _make_terminal(result="base")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert result == "base_async"


@pytest.mark.asyncio
async def test_output_does_not_affect_events(registry, stage):
    def output_handler(result):
        return "transformed"

    registry.add_output(stage.Output, output_handler)
    terminal = _make_terminal("e1", "e2", result="original")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert events == ["e1", "e2"]
    assert result == "transformed"


@pytest.mark.asyncio
async def test_output_runs_after_wrap(registry, stage):
    order: list[str] = []

    async def wrap_handler(context, next_fn):
        order.append("wrap_before")
        async for event in next_fn(context):
            yield event
        order.append("wrap_after")

    def output_handler(result):
        order.append("output")
        return result + "_out"

    registry.add_output(stage.Output, output_handler)
    registry.add(stage, wrap_handler)
    terminal = _make_terminal(result="base")
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert "output" in order
    assert result == "base_out"


# --- remove ---


@pytest.mark.asyncio
async def test_remove_handler(registry, stage):
    called = False

    async def handler(context, next_fn):
        nonlocal called
        called = True
        async for event in next_fn(context):
            yield event

    registry.add(stage, handler)
    registry.remove(stage, handler)
    terminal = _make_terminal(result="done")
    chain = registry.compose(stage, terminal)
    assert chain is terminal
    events, result = await _collect(registry.invoke(stage, {}, terminal))
    assert not called
    assert result == "done"


@pytest.mark.asyncio
async def test_remove_input_adapted_handler(registry, stage):
    def input_handler(context):
        return {**context, "should_not_appear": True}

    adapted = registry.add_input(stage.Input, input_handler)
    registry.remove(stage.Input._stage, adapted)
    received = {}

    async def terminal(context):
        received.update(context)
        yield _MiddlewareResult("done")

    await _collect(registry.invoke(stage, {"original": True}, terminal))
    assert "should_not_appear" not in received


@pytest.mark.asyncio
async def test_remove_nonexistent_is_noop(registry, stage):
    async def handler(context, next_fn):
        async for event in next_fn(context):
            yield event

    registry.remove(stage, handler)


# --- error propagation ---


@pytest.mark.asyncio
async def test_terminal_error_propagates_through_middleware(registry, stage):
    async def terminal(context):
        raise ValueError("terminal_error")
        yield  # noqa: unreachable — makes it a generator

    async def passthrough(context, next_fn):
        async for event in next_fn(context):
            yield event

    registry.add(stage, passthrough)
    with pytest.raises(ValueError, match="terminal_error"):
        await _collect(registry.invoke(stage, {}, terminal))


@pytest.mark.asyncio
async def test_middleware_error_propagates_to_caller(registry, stage):
    async def broken_middleware(context, next_fn):
        raise RuntimeError("middleware_error")
        yield  # noqa: unreachable

    registry.add(stage, broken_middleware)
    terminal = _make_terminal(result="done")
    with pytest.raises(RuntimeError, match="middleware_error"):
        await _collect(registry.invoke(stage, {}, terminal))


@pytest.mark.asyncio
async def test_try_finally_in_middleware_runs_on_error(registry, stage):
    finally_ran = False

    async def terminal(context):
        raise ValueError("boom")
        yield  # noqa: unreachable

    async def guarded(context, next_fn):
        nonlocal finally_ran
        try:
            async for event in next_fn(context):
                yield event
        finally:
            finally_ran = True

    registry.add(stage, guarded)
    with pytest.raises(ValueError, match="boom"):
        await _collect(registry.invoke(stage, {}, terminal))
    assert finally_ran


@pytest.mark.asyncio
async def test_try_finally_runs_on_generator_close(registry, stage):
    finally_ran = False

    async def guarded(context, next_fn):
        nonlocal finally_ran
        try:
            async for event in next_fn(context):
                yield event
        finally:
            finally_ran = True

    registry.add(stage, guarded)
    terminal = _make_terminal("e1", "e2", "e3", result="done")
    gen = registry.invoke(stage, {}, terminal)
    await gen.__anext__()
    await gen.aclose()
    assert finally_ran


# --- additional coverage ---


@pytest.mark.asyncio
async def test_chained_context_modification_across_wrap_handlers(registry, stage):
    """Multiple Wrap handlers each modify context; terminal sees accumulated changes."""
    received_context = {}

    async def terminal(context):
        received_context.update(context)
        yield _MiddlewareResult("done")

    async def add_a(context, next_fn):
        async for event in next_fn({**context, "a": True}):
            yield event

    async def add_b(context, next_fn):
        async for event in next_fn({**context, "b": True}):
            yield event

    registry.add(stage, add_a)
    registry.add(stage, add_b)
    await _collect(registry.invoke(stage, {"original": True}, terminal))
    assert received_context == {"original": True, "a": True, "b": True}


@pytest.mark.asyncio
async def test_error_transformation_by_middleware(registry, stage):
    """Middleware can catch and re-throw a different error."""

    async def terminal(context):
        raise ValueError("original")
        yield  # noqa: unreachable

    async def transformer(context, next_fn):
        try:
            async for event in next_fn(context):
                yield event
        except ValueError as e:
            raise RuntimeError(f"Wrapped: {e}") from e

    registry.add(stage, transformer)
    with pytest.raises(RuntimeError, match="Wrapped: original"):
        await _collect(registry.invoke(stage, {}, terminal))


@pytest.mark.asyncio
async def test_interrupt_exception_propagates_through_passthrough(registry, stage):
    """InterruptException propagates through passthrough middleware without being swallowed."""
    from strands.interrupt import Interrupt, InterruptException

    async def terminal(context):
        raise InterruptException(Interrupt(id="int-1", name="test"))
        yield  # noqa: unreachable

    async def passthrough(context, next_fn):
        async for event in next_fn(context):
            yield event

    registry.add(stage, passthrough)
    with pytest.raises(InterruptException):
        await _collect(registry.invoke(stage, {}, terminal))


@pytest.mark.asyncio
async def test_multi_layer_finally_ordering_on_error(registry, stage):
    """All finally blocks run in reverse order (inner first) when terminal throws."""
    order: list[str] = []

    async def terminal(context):
        raise ValueError("boom")
        yield  # noqa: unreachable

    async def outer(context, next_fn):
        try:
            async for event in next_fn(context):
                yield event
        finally:
            order.append("outer_finally")

    async def inner(context, next_fn):
        try:
            async for event in next_fn(context):
                yield event
        finally:
            order.append("inner_finally")

    registry.add(stage, outer)
    registry.add(stage, inner)
    with pytest.raises(ValueError, match="boom"):
        await _collect(registry.invoke(stage, {}, terminal))
    assert order == ["inner_finally", "outer_finally"]


@pytest.mark.asyncio
async def test_two_layer_finally_on_generator_close(registry, stage):
    """Both middleware finally blocks run when consumer calls aclose."""
    order: list[str] = []

    async def outer(context, next_fn):
        try:
            async for event in next_fn(context):
                yield event
        finally:
            order.append("outer_finally")

    async def inner(context, next_fn):
        try:
            async for event in next_fn(context):
                yield event
        finally:
            order.append("inner_finally")

    registry.add(stage, outer)
    registry.add(stage, inner)
    terminal = _make_terminal("e1", "e2", "e3", result="done")
    gen = registry.invoke(stage, {}, terminal)
    await gen.__anext__()
    await gen.aclose()
    assert "inner_finally" in order
    assert "outer_finally" in order


@pytest.mark.asyncio
async def test_remove_only_first_occurrence_of_duplicate(registry, stage):
    """Removing a handler registered twice only removes one occurrence."""
    call_count = 0

    async def handler(context, next_fn):
        nonlocal call_count
        call_count += 1
        async for event in next_fn(context):
            yield event

    registry.add(stage, handler)
    registry.add(stage, handler)
    registry.remove(stage, handler)

    terminal = _make_terminal(result="done")
    await _collect(registry.invoke(stage, {}, terminal))
    assert call_count == 1
