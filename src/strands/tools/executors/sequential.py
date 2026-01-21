"""Sequential tool executor implementation."""

from typing import TYPE_CHECKING, Any, AsyncGenerator

from typing_extensions import override

from ...hooks import AfterToolsEvent, BeforeToolsEvent
from ...telemetry.metrics import Trace
from ...types._events import ToolInterruptEvent, ToolsInterruptEvent, TypedEvent
from ...types.content import Message
from ...types.tools import ToolResult, ToolUse
from ._executor import ToolExecutor

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent
    from ..structured_output._structured_output_context import StructuredOutputContext


class SequentialToolExecutor(ToolExecutor):
    """Sequential tool executor."""

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
        """Execute tools sequentially.

        Breaks early if an interrupt is raised by the user.

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
        # Skip batch events if no tools to execute
        if not tool_uses:
            return
        
        # Trigger BeforeToolsEvent
        before_event = BeforeToolsEvent(agent=agent, message=message, tool_uses=tool_uses)
        _, interrupts = await agent.hooks.invoke_callbacks_async(before_event)

        if interrupts:
            # Use ToolsInterruptEvent for batch-level interrupts
            yield ToolsInterruptEvent(tool_uses, interrupts)
            # Always fire AfterToolsEvent even if interrupted
            after_event = AfterToolsEvent(agent=agent, message=message, tool_uses=tool_uses)
            await agent.hooks.invoke_callbacks_async(after_event)
            return

        interrupted = False

        for tool_use in tool_uses:
            events = ToolExecutor._stream_with_trace(
                agent, tool_use, tool_results, cycle_trace, cycle_span, invocation_state, structured_output_context
            )
            async for event in events:
                if isinstance(event, ToolInterruptEvent):
                    interrupted = True

                yield event

            if interrupted:
                break

        # Always trigger AfterToolsEvent
        after_event = AfterToolsEvent(agent=agent, message=message, tool_uses=tool_uses)
        await agent.hooks.invoke_callbacks_async(after_event)
