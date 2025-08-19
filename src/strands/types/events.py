from dataclasses import MISSING
from typing import Any, AsyncGenerator, Callable, Literal, override

from ..telemetry import EventLoopMetrics
from .content import Message
from .event_loop import StopReason
from .tools import ToolResult, ToolUse


class TypedEvent(dict):
    # Do I have to be careful about tool stream results?
    invocation_state: dict[str, Any]

    def __init__(self, data: dict[str, Any] = None):
        super().__init__(data or {})

    def _get_callback_fields(self) -> list[str] | Literal["all"] | None:
        return None

    def set_invocation_state(self, invocation_state: dict):
        self.invocation_state = invocation_state

    def prepare_and_invoke(self, invocation_state: dict, callback_handler: Callable):
        self.set_invocation_state(invocation_state)

        # invoking
        args = TypedEvent.get_callback_fields(self)

        if args:
            callback_handler(**args)
            return True

        return False

    @staticmethod
    def get_callback_fields(event: "TypedEvent") -> dict[str, Any] | None:
        allow_listed = event._get_callback_fields()

        if allow_listed is None:
            return None
        else:
            if allow_listed is Any:
                return {**event}
            else:
                return {k: v for k, v in event.items() if k in allow_listed}


class StartEventLoopEvent(TypedEvent):
    def __init__(self):
        super().__init__({"start_event_loop": True})

    def _get_callback_fields(self):
        return ["start_event_loop"]


class InitEventLoopEvent(TypedEvent):
    def __init__(self):
        super().__init__({"init_event_loop": True})

    @override
    def set_invocation_state(self, invocation_state: dict):
        super().set_invocation_state(invocation_state)

        # For backwards compatability, make sure that we're merging the
        # invocation state as a readonly copy into ourselves
        self.update(**invocation_state)

    def _get_callback_fields(self):
        return ["init_event_loop"]


class ForceStopEvent(TypedEvent):
    @property
    def reason(self) -> str:
        return self.get("force_stop_reason")

    @property
    def reason_exception(self) -> Exception | None:
        return self.get("force_stop_reason_exception")

    def __init__(self, reason: str | Exception):
        super().__init__(
            {"force_stop": True, "force_stop_reason": str(reason), "force_stop_reason_exception": reason or MISSING}
        )

    def _get_callback_fields(self):
        return ["force_stop", "force_stop_reason"]


class MessageEvent(TypedEvent):
    def __init__(self, message: Message):
        super().__init__({"message": message})

    @property
    def message(self) -> Message:
        return self.get("message")

    def _get_callback_fields(self):
        return ["message"]


class EventLoopThrottleDelay(TypedEvent):
    def __init__(self, delay: int):
        super().__init__({"event_loop_throttled_delay": delay})

    @property
    def delay(self) -> Message:
        return self.get("event_loop_throttled_delay")

    def _get_callback_fields(self):
        return ["event_loop_throttled_delay"]


class StopEvent(TypedEvent):
    def __init__(
        self,
        stop_reason: StopReason,
        message: Message,
        metrics: "EventLoopMetrics",
        request_state: Any,
    ):
        super().__init__(
            {
                "stop": (stop_reason, message, metrics, request_state),
            }
        )

    @property
    def stop_reason(self) -> StopReason:
        return self.get("stop")[0]

    @property
    def message(self) -> Message:
        return self.get("stop")[1]

    @property
    def metrics(self) -> "EventLoopMetrics":
        return self.get("stop")[2]

    @property
    def request_state(self) -> Any:
        return self.get("stop")[3]

    @property
    def result(self) -> "AgentResult":
        from strands.agent import AgentResult

        return AgentResult(
            stop_reason=self.stop_reason,
            message=self.message,
            metrics=self.metrics,
            state=self.request_state,
        )

    def _get_callback_fields(self):
        return ["stop"]


class StartEvent(TypedEvent):
    def __init__(self):
        super().__init__({"start": True})

    def _get_callback_fields(self):
        return ["start"]


class ToolStreamEvent(TypedEvent):
    def __init__(self, tool_use: ToolUse, stream_data: any):
        super().__init__({"tool_stream_tool_use": tool_use, "tool_stream_data": stream_data})

    @property
    def tool_use(self) -> ToolUse:
        return self.get("tool_stream_tool_use")

    @property
    def tool_stream_data(self) -> Any:
        return self.get("tool_stream_data")


class ToolResultEvent(TypedEvent):
    def __init__(self, tool_result: ToolResult):
        super().__init__({"tool_result": tool_result})

    @property
    def tool_result(self) -> ToolResult:
        return self.get("tool_result")


class ToolResultMessageEvent(TypedEvent):
    def __init__(self, message: Any):
        super().__init__({"message": message})

    @property
    def message(self) -> Any:
        return self.get("message")

    @override
    def _get_callback_fields(self):
        return ["message"]


class StreamDeltaEvent(TypedEvent):
    def __init__(self, delta_data: dict[str, Any]):
        super().__init__(delta_data)

    @property
    def delta_data(self) -> Any:
        return self

    def include_state(self):
        return True

    def as_callback(self) -> dict[str, Any]:
        return self.delta_data

    def set_invocation_state(self, invocation_state: dict):
        super().set_invocation_state(invocation_state)

        # For backwards compatability, make sure that we're merging the
        # invocation state as a readonly copy into ourselves
        if "delta" in self.delta_data:
            self.update(**invocation_state)

    @override
    def _get_callback_fields(self):
        return Any


class ResultEvent(TypedEvent):
    def __init__(self, result: "AgentResult"):
        super().__init__({"result": result})

    @property
    def result(self) -> "AgentResult":
        return self.get("result")

    @override
    def _get_callback_fields(self):
        return ["result"]


TypedToolGenerator = AsyncGenerator[TypedEvent, None]
"""Generator of tool events where all events are typed as TypedEvents."""
