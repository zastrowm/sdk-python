"""Tool execution functionality for the event loop."""

import logging
import time
from typing import Callable, List, Optional, Tuple

from opentelemetry import trace

from ..telemetry.metrics import EventLoopMetrics, Trace
from ..telemetry.tracer import get_tracer
from ..tools.tools import InvalidToolUseNameException, validate_tool_use
from ..types.content import Message
from ..types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)


def run_single_tool(
    tool: ToolUse,
    handler: Callable[[ToolUse], ToolResult],
    event_loop_metrics: EventLoopMetrics,
    invalid_tool_use_ids: List[str],
    cycle_trace: Trace,
    parent_span: Optional[trace.Span] = None,
) -> Tuple[bool, Optional[ToolResult]]:
    result = None
    tool_succeeded = False

    tracer = get_tracer()
    tool_call_span = tracer.start_tool_call_span(tool, parent_span)

    try:
        if "toolUseId" not in tool or tool["toolUseId"] not in invalid_tool_use_ids:
            tool_name = tool["name"]
            tool_trace = Trace(f"Tool: {tool_name}", parent_id=cycle_trace.id, raw_name=tool_name)
            tool_start_time = time.time()
            result = handler(tool)
            tool_success = result.get("status") == "success"
            if tool_success:
                tool_succeeded = True

            tool_duration = time.time() - tool_start_time
            message = Message(role="user", content=[{"toolResult": result}])
            event_loop_metrics.add_tool_usage(tool, tool_duration, tool_trace, tool_success, message)
            cycle_trace.add_child(tool_trace)

        if tool_call_span:
            tracer.end_tool_call_span(tool_call_span, result)
    except Exception as e:
        if tool_call_span:
            tracer.end_span_with_error(tool_call_span, str(e), e)

    return tool_succeeded, result


def validate_and_prepare_tools(
    message: Message,
) -> Tuple[List[ToolUse], List[ToolResult], List[str]]:
    """Validate tool uses and prepare them for execution.

    Args:
        message: Current message.
        tool_uses: List to populate with tool uses.
        tool_results: List to populate with tool results for invalid tools.
        invalid_tool_use_ids: List to populate with invalid tool use IDs.
    """

    tool_uses: List[ToolUse] = []
    tool_results: List[ToolResult] = []
    invalid_tool_use_ids: List[str] = []

    # Extract tool uses from the message
    for content in message["content"]:
        if isinstance(content, dict) and "toolUse" in content:
            tool_uses.append(content["toolUse"])

    # Validate tool uses
    # Avoid modifying original `tool_uses` variable during iteration
    tool_uses_copy = tool_uses.copy()
    for tool in tool_uses_copy:
        try:
            validate_tool_use(tool)
        except InvalidToolUseNameException as e:
            # Replace the invalid toolUse name and return an invalid name error as ToolResult to the LLM as context
            tool_uses.remove(tool)
            tool["name"] = "INVALID_TOOL_NAME"
            invalid_tool_use_ids.append(tool["toolUseId"])
            tool_uses.append(tool)
            tool_results.append(
                {
                    "toolUseId": tool["toolUseId"],
                    "status": "error",
                    "content": [{"text": f"Error: {str(e)}"}],
                }
            )

    return tool_uses, tool_results, invalid_tool_use_ids