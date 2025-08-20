"""Event system for the Strands Agents framework.

This module defines the event types that are emitted during agent execution,
providing a structured way to observe and respond to different stages of the
agent lifecycle including initialization, model invocation, tool execution,
and completion.
"""

from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Literal

from typing_extensions import override

from ..telemetry import EventLoopMetrics
from .content import Message
from .event_loop import StopReason
from .streaming import ContentBlockDelta
from .tools import ToolResult, ToolUse

if TYPE_CHECKING:
    from ..agent import Agent, AgentResult


class TypedEvent(dict):
    """Base class for all typed events in the agent system.

    TypedEvent provides a framework for creating strongly-typed events that can be
    processed by callback handlers.
    """

    invocation_state: dict[str, Any]

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        """Initialize the typed event with optional data.

        Args:
            data: Optional dictionary of event data to initialize with
        """
        super().__init__(data or {})

    def _get_callback_fields(self) -> list[str] | Literal["all"] | None:
        """When invoking a callback with this event, which fields should be exposed.

        This is for **backwards compatability only** and should not be implemented for new events.

        Can be an array of properties to pass to the callback handler, can be the literal "all" to
        include all properties in the dict, or None to indicate that the callback should not be invoked.
        """
        return None

    def _set_invocation_state(self, invocation_state: dict) -> None:
        """Sets the invocation state for this event instance.

        This is for **backwards compatability only** and should not be implemented for new events.

        Args:
            invocation_state: Dictionary containing context and state information
                for the current agent invocation
        """
        self.invocation_state = invocation_state

    def _prepare_and_invoke(self, *, agent: "Agent", invocation_state: dict, callback_handler: Callable) -> bool:
        """Internal API: Prepares the event and invokes the callback handler if applicable.

        This method sets the invocation state on the event and invokes the callback_handler with
        the event if it is configured to be invoked

        Args:
            agent: The agent instance (not currently used but will be in future versions)
            invocation_state: Context and state information for the current invocation
            callback_handler: The callback function to invoke with event data

        Returns:
            bool: True if the callback was invoked, False otherwise
        """
        self._set_invocation_state(invocation_state)
        args = TypedEvent._get_callback_arguments(self)

        if args:
            callback_handler(**args)
            return True

        return False

    @staticmethod
    def _get_callback_arguments(event: "TypedEvent") -> dict[str, Any] | None:
        allow_listed = event._get_callback_fields()

        if allow_listed is None:
            return None
        else:
            if allow_listed is Any:
                return {**event}
            else:
                return {k: v for k, v in event.items() if k in allow_listed}


class InitEventLoopEvent(TypedEvent):
    """Event emitted at the very beginning of agent execution.

    This event is fired before any processing begins and provides access to the
    initial invocation state.
    """

    def __init__(self) -> None:
        """Initialize the event loop initialization event."""
        super().__init__({"init_event_loop": True})

    @override
    def _set_invocation_state(self, invocation_state: dict) -> None:
        super()._set_invocation_state(invocation_state)

        # For backwards compatability, make sure that we're merging the
        # invocation state as a readonly copy into ourselves
        self.update(**invocation_state)

    def _get_callback_fields(self) -> list[str]:
        return ["init_event_loop"]


class StartEvent(TypedEvent):
    """Event emitted at the start of each event loop cycle.

    ::deprecated::
        Use StartEventLoopEvent instead.

    This event signals the beginning of a new processing cycle within the agent's
    event loop. It's fired before model invocation and tool execution begin.
    """

    def __init__(self) -> None:
        """Initialize the event loop start event."""
        super().__init__({"start": True})

    def _get_callback_fields(self) -> list[str]:
        return ["start"]


class StartEventLoopEvent(TypedEvent):
    """Event emitted when the event loop cycle begins processing.

    This event is fired after StartEvent and indicates that the event loop
    has begun its core processing logic, including model invocation preparation.
    """

    def __init__(self) -> None:
        """Initialize the event loop processing start event."""
        super().__init__({"start_event_loop": True})

    def _get_callback_fields(self) -> list[str]:
        return ["start_event_loop"]


class MessageEvent(TypedEvent):
    """Event emitted when the model invocation has completed.

    This event is fired whenever the model generates a response message that
    gets added to the conversation history.
    """

    def __init__(self, message: Message) -> None:
        """Initialize with the model-generated message.

        Args:
            message: The response message from the model
        """
        super().__init__({"message": message})

    @property
    def message(self) -> Message:
        """The model-generated response message."""
        return self.get("message")  # type: ignore[return-value]

    def _get_callback_fields(self) -> list[str]:
        return ["message"]


class EventLoopThrottleDelay(TypedEvent):
    """Event emitted when the event loop is throttled due to rate limiting."""

    def __init__(self, delay: int) -> None:
        """Initialize with the throttle delay duration.

        Args:
            delay: Delay in seconds before the next retry attempt
        """
        super().__init__({"event_loop_throttled_delay": delay})

    @property
    def delay(self) -> int:
        """Delay in seconds before the next retry attempt."""
        return self.get("event_loop_throttled_delay")  # type: ignore[return-value]

    def _get_callback_fields(self) -> list[str]:
        return ["event_loop_throttled_delay"]


class ForceStopEvent(TypedEvent):
    """Event emitted when the agent execution is forcibly stopped, either by a tool or by an exception.

    This event is fired when an unrecoverable error occurs during execution,
    such as repeated throttling failures or critical system errors. It provides
    the reason for the forced stop and any associated exception.
    """

    @property
    def reason(self) -> str:
        """Human-readable description of why execution was stopped."""
        return self.get("force_stop_reason")  # type: ignore[return-value]

    @property
    def reason_exception(self) -> Exception | None:
        """The original exception that caused the stop, if applicable."""
        return self.get("force_stop_reason_exception")

    def __init__(self, reason: str | Exception) -> None:
        """Initialize with the reason for forced stop.

        Args:
            reason: String description or exception that caused the forced stop
        """
        super().__init__(
            {
                "force_stop": True,
                "force_stop_reason": str(reason),
                "force_stop_reason_exception": reason if reason and isinstance(reason, Exception) else MISSING,
            }
        )

    def _get_callback_fields(self) -> list[str]:
        return ["force_stop", "force_stop_reason"]


class StopEvent(TypedEvent):
    """Event emitted when the agent execution completes normally.

    This event is fired when the event loop cycle completes successfully,
    providing the final result including the stop reason, final message,
    execution metrics, and final state. It represents the successful
    completion of an agent invocation.
    """

    def __init__(
        self,
        stop_reason: StopReason,
        message: Message,
        metrics: "EventLoopMetrics",
        request_state: Any,
    ) -> None:
        """Initialize with the final execution results.

        Args:
            stop_reason: Why the agent execution stopped
            message: Final message from the model
            metrics: Execution metrics and performance data
            request_state: Final state of the agent execution
        """
        from ..agent import AgentResult

        super().__init__(
            {
                "result": AgentResult(
                    stop_reason=stop_reason,
                    message=message,
                    metrics=metrics,
                    state=request_state,
                ),
            }
        )

    @property
    def result(self) -> "AgentResult":
        """Complete execution result with metrics and final state."""
        return self.get("result")  # type: ignore[return-value]

    @override
    def _get_callback_fields(self) -> list[str]:
        return ["result"]


class ToolStreamEvent(TypedEvent):
    """Event emitted during tool execution streaming.

    This event is fired when a tool produces streaming output during execution.
    It provides access to the tool use information and the streaming data,
    allowing callbacks to process tool execution progress in real-time.
    """

    def __init__(self, tool_use: ToolUse, stream_data: Any) -> None:
        """Initialize with tool streaming data.

        Args:
            tool_use: The tool invocation producing the stream
            stream_data: Incremental data from the tool execution
        """
        super().__init__({"tool_stream_tool_use": tool_use, "tool_stream_data": stream_data})

    @property
    def tool_use(self) -> ToolUse:
        """The tool invocation that is producing streaming output."""
        return self.get("tool_stream_tool_use")  # type: ignore[return-value]

    @property
    def tool_stream_data(self) -> Any:
        """Incremental data chunk from the streaming tool execution."""
        return self.get("tool_stream_data")


class ToolResultEvent(TypedEvent):
    """Event emitted when a tool execution completes."""

    def __init__(self, tool_result: ToolResult) -> None:
        """Initialize with the completed tool result.

        Args:
            tool_result: Final result from the tool execution
        """
        super().__init__({"tool_result": tool_result})

    @property
    def tool_result(self) -> ToolResult:
        """Final result from the completed tool execution."""
        return self.get("tool_result")  # type: ignore[return-value]


class ToolResultMessageEvent(TypedEvent):
    """Event emitted when tool results are formatted as a message.

    This event is fired when tool execution results are converted into a
    message format to be added to the conversation history. It provides
    access to the formatted message containing tool results.
    """

    def __init__(self, message: Any) -> None:
        """Initialize with the formatted tool result message.

        Args:
            message: Message containing tool results for conversation history
        """
        super().__init__({"message": message})

    @property
    def message(self) -> Any:
        """Message containing formatted tool results for conversation history."""
        return self.get("message")

    @override
    def _get_callback_fields(self) -> list[str]:
        return ["message"]


class ModelStreamEvent(TypedEvent):
    """Event emitted during model response streaming.

    This event is fired when the model produces streaming output during response
    generation. It provides access to incremental delta data from the model,
    allowing callbacks to process and display streaming responses in real-time.
    """

    def __init__(self, delta_data: dict[str, Any]) -> None:
        """Initialize with streaming delta data from the model.

        Args:
            delta_data: Incremental streaming data from the model response
        """
        super().__init__(delta_data)

    @property
    def delta_data(self) -> Any:
        """Streaming data from the model response."""
        return self

    @override
    def _set_invocation_state(self, invocation_state: dict) -> None:
        super()._set_invocation_state(invocation_state)

        # For backwards compatability, make sure that we're merging the
        # invocation state as a readonly copy into ourselves
        if "delta" in self.delta_data:
            self.update(**invocation_state)

    @override
    def _get_callback_fields(self) -> Any:
        return Any


class ToolUseStreamEvent(ModelStreamEvent):
    """Event emitted during tool use input streaming."""

    def __init__(self, delta: ContentBlockDelta, current_tool_use: dict[str, Any]) -> None:
        """Initialize with delta and current tool use state."""
        super().__init__({"delta": delta, "current_tool_use": current_tool_use})

    @property
    def delta(self) -> ContentBlockDelta:
        """Delta data from the model response."""
        return self.get("delta")  # type: ignore[return-value]

    @property
    def current_tool_use(self) -> dict[str, Any]:
        """Current tool use state."""
        return self.get("current_tool_use")  # type: ignore[return-value]


class TextStreamEvent(ModelStreamEvent):
    """Event emitted during text content streaming."""

    def __init__(self, delta: ContentBlockDelta, text: str) -> None:
        """Initialize with delta and text content."""
        super().__init__({"data": text, "delta": delta})

    @property
    def text(self) -> str:
        """Cumulative text content assembled from streaming deltas received thus far."""
        return self.get("data")  # type: ignore[return-value]

    @property
    def delta(self) -> ContentBlockDelta:
        """Delta data from the model response."""
        return self.get("delta")  # type: ignore[return-value]


class ReasoningTextStreamEvent(ModelStreamEvent):
    """Event emitted during reasoning text streaming."""

    def __init__(self, delta: ContentBlockDelta, reasoning_text: str | None) -> None:
        """Initialize with delta and reasoning text."""
        super().__init__({"reasoningText": reasoning_text, "delta": delta, "reasoning": True})

    @property
    def reasoningText(self) -> str:
        """Cumulative reasoning text content assembled from streaming deltas received thus far."""
        return self.get("reasoningText")  # type: ignore[return-value]

    @property
    def delta(self) -> ContentBlockDelta:
        """Delta data from the model response."""
        return self.get("delta")  # type: ignore[return-value]


class ReasoningSignatureStreamEvent(ModelStreamEvent):
    """Event emitted during reasoning signature streaming."""

    def __init__(self, delta: ContentBlockDelta, reasoning_signature: str | None) -> None:
        """Initialize with delta and reasoning signature."""
        super().__init__({"reasoning_signature": reasoning_signature, "delta": delta, "reasoning": True})

    @property
    def reasoning_signature(self) -> str:
        """Cumulative reasoning signature content assembled from streaming deltas received thus far."""
        return self.get("reasoning_signature")  # type: ignore[return-value]

    @property
    def delta(self) -> ContentBlockDelta:
        """Delta data from the model response."""
        return self.get("delta")  # type: ignore[return-value]


AllTypedEvents = (
    InitEventLoopEvent
    | StartEvent
    | StartEventLoopEvent
    | MessageEvent
    | EventLoopThrottleDelay
    | ForceStopEvent
    | StopEvent
    | ToolStreamEvent
    | ToolResultEvent
    | ToolResultMessageEvent
    | ModelStreamEvent
    | ToolUseStreamEvent
    | TextStreamEvent
    | ReasoningTextStreamEvent
    | ReasoningSignatureStreamEvent
    | TypedEvent
)

TypedToolGenerator = AsyncGenerator[TypedEvent, None]
"""Generator of tool events where all events are typed as TypedEvents."""
