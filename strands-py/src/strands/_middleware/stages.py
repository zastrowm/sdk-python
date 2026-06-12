"""Built-in middleware stages and their context/result types."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..interrupt import Interrupt, InterruptException

from .types import MiddlewareStage

if TYPE_CHECKING:
    from ..agent.agent import Agent
    from ..interrupt import _InterruptState
    from ..types.content import Messages, SystemPrompt
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
    system_prompt: SystemPrompt
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
class MiddlewareInterruptResult:
    """Result returned by interrupt() in middleware contexts.

    Wraps the response in an object to allow future extension (e.g., cached data,
    metadata) without a breaking change to callers.
    """

    response: Any


@dataclass
class ExecuteToolContext:
    """Context passed to ExecuteToolStage middleware.

    Contains everything needed to understand and potentially modify the tool call.
    Supports middleware-initiated interrupts via the `interrupt()` method.
    """

    agent: Agent
    tool: AgentTool | None
    tool_use: ToolUse
    invocation_state: dict[str, Any]
    _interrupt_state: _InterruptState = field(repr=False, default=None)  # type: ignore[assignment]

    def interrupt(self, name: str, *, reason: Any = None, response: Any = None) -> MiddlewareInterruptResult:
        """Request a human-in-the-loop interrupt.

        On first execution (no prior response), raises InterruptException to halt the agent.
        On resume (after the user provides a response), returns the response wrapped in
        MiddlewareInterruptResult.

        Args:
            name: User-defined name for the interrupt. Must be unique within this middleware.
            reason: Optional reason for the interrupt (shown to the user).
            response: Optional preemptive response (skips the interrupt if provided).

        Returns:
            MiddlewareInterruptResult containing the user's response.

        Raises:
            InterruptException: When no response has been provided yet.
        """
        interrupt_id = f"middleware:executeTool:{self.tool_use['toolUseId']}:{uuid.uuid5(uuid.NAMESPACE_OID, name)}"
        state = self._interrupt_state

        existing = state.interrupts.get(interrupt_id)
        if existing and existing.response is not None:
            return MiddlewareInterruptResult(response=existing.response)

        if response is not None:
            return MiddlewareInterruptResult(response=response)

        interrupt = Interrupt(id=interrupt_id, name=name, reason=reason)
        state.interrupts.setdefault(interrupt_id, interrupt)
        raise InterruptException(interrupt)


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
