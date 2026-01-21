"""Concurrent tool executor implementation."""

import asyncio
from typing import TYPE_CHECKING, Any, AsyncGenerator

from typing_extensions import override

from ...hooks import AfterToolsEvent, BeforeToolsEvent
from ...telemetry.metrics import Trace
from ...types._events import ToolInterruptEvent, TypedEvent
from ...types.content import Message
from ...types.tools import ToolResult, ToolUse
from ._executor import ToolExecutor

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent
    from ..structured_output._structured_output_context import StructuredOutputContext


class ConcurrentToolExecutor(ToolExecutor):
    """Concurrent tool executor."""

    @override
    async def _execute(
        self,
        agent: "Agent",
        message: Message,
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: "StructuredOutputContext | None" = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute tools concurrently.

        Args:
            agent: The agent for which tools are being executed.
            message: The message from the model containing tool use blocks.
            tool_uses: Metadata and inputs for the tools to be executed.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for the tool invocation.
            structured_output_context: Context for structured output handling.

        Yields:
            Events from the tool execution stream.
        """
        # Skip batch events if no tools
        if not tool_uses:
            return

        # Trigger BeforeToolsEvent
        before_event = BeforeToolsEvent(agent=agent, message=message, tool_uses=tool_uses)
        _, interrupts = await agent.hooks.invoke_callbacks_async(before_event)

        if interrupts:
            # Use the first tool_use for the interrupt event (tools not executed yet)
            yield ToolInterruptEvent(tool_uses[0], interrupts)
            return

        task_queue: asyncio.Queue[tuple[int, Any]] = asyncio.Queue()
        task_events = [asyncio.Event() for _ in tool_uses]
        stop_event = object()

        tasks = [
            asyncio.create_task(
                self._task(
                    agent,
                    tool_use,
                    tool_results,
                    cycle_trace,
                    cycle_span,
                    invocation_state,
                    task_id,
                    task_queue,
                    task_events[task_id],
                    stop_event,
                    structured_output_context,
                )
            )
            for task_id, tool_use in enumerate(tool_uses)
        ]

        task_count = len(tasks)
        while task_count:
            task_id, event = await task_queue.get()
            if event is stop_event:
                task_count -= 1
                continue

            yield event
            task_events[task_id].set()

        # Trigger AfterToolsEvent
        after_event = AfterToolsEvent(agent=agent, message=message, tool_uses=tool_uses)
        await agent.hooks.invoke_callbacks_async(after_event)

    async def _task(
        self,
        agent: "Agent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        task_id: int,
        task_queue: asyncio.Queue,
        task_event: asyncio.Event,
        stop_event: object,
        structured_output_context: "StructuredOutputContext | None",
    ) -> None:
        """Execute a single tool and put results in the task queue.

        Args:
            agent: The agent executing the tool.
            tool_use: Tool use metadata and inputs.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for tool execution.
            task_id: Unique identifier for this task.
            task_queue: Queue to put tool events into.
            task_event: Event to signal when task can continue.
            stop_event: Sentinel object to signal task completion.
            structured_output_context: Context for structured output handling.
        """
        try:
            events = ToolExecutor._stream_with_trace(
                agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
            )
            async for event in events:
                task_queue.put_nowait((task_id, event))
                await task_event.wait()
                task_event.clear()

        finally:
            task_queue.put_nowait((task_id, stop_event))
