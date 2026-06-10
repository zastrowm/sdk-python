"""Built-in middleware stages and their context/result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .types import MiddlewareStage

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..types.content import Messages
    from ..types.streaming import StopReason, Usage
    from ..types.tools import AgentTool, ToolSpec, ToolUse


@dataclass
class InvokeModelContext:
    """Context passed to InvokeModelStage middleware.

    All fields represent the inputs to the model call. Middleware can inspect
    or transform any of them by passing a modified context to next().
    """

    agent: Agent
    messages: Messages
    system_prompt: str | None
    system_prompt_content: list[Any] | None
    tool_specs: list[ToolSpec]
    tool_choice: Any | None
    invocation_state: dict[str, Any]
    model_state: dict[str, Any]


@dataclass
class InvokeModelResult:
    """Result from InvokeModelStage middleware.

    Contains the aggregated result of a model call.
    """

    stop_reason: StopReason
    message: dict[str, Any]
    usage: dict[str, Any]
    metrics: dict[str, Any]


InvokeModelStage: MiddlewareStage[InvokeModelContext, InvokeModelResult, Any] = MiddlewareStage(name="invokeModel")
"""Built-in stage wrapping core model invocation.

Middleware registered for this stage can rate-limit, cache, or transform model inputs/outputs.
"""


@dataclass
class ExecuteToolContext:
    """Context passed to ExecuteToolStage middleware.

    Contains everything needed to understand and potentially modify the tool call.
    """

    agent: Agent
    tool: AgentTool | None
    tool_use: ToolUse
    invocation_state: dict[str, Any]


@dataclass
class ExecuteToolResult:
    """Result from ExecuteToolStage middleware.

    Contains the tool result and optional exception from execution.
    """

    tool_result: dict[str, Any]
    exception: Exception | None = None


ExecuteToolStage: MiddlewareStage[ExecuteToolContext, ExecuteToolResult, Any] = MiddlewareStage(name="executeTool")
"""Built-in stage wrapping individual tool execution.

Middleware registered for this stage can add telemetry, validate inputs, or mock responses.
"""
