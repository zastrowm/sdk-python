"""Event loop-related type definitions for the SDK."""
from dataclasses import dataclass
from typing import Literal, Any, TYPE_CHECKING, Callable

from typing_extensions import TypedDict


from .content import Message
from .tools import ToolResult

if TYPE_CHECKING:
    from ..telemetry import EventLoopMetrics


class Usage(TypedDict):
    """Token usage information for model interactions.

    Attributes:
        inputTokens: Number of tokens sent in the request to the model..
        outputTokens: Number of tokens that the model generated for the request.
        totalTokens: Total number of tokens (input + output).
    """

    inputTokens: int
    outputTokens: int
    totalTokens: int


class Metrics(TypedDict):
    """Performance metrics for model interactions.

    Attributes:
        latencyMs (int): Latency of the model request in milliseconds.
    """

    latencyMs: int


StopReason = Literal[
    "content_filtered",
    "end_turn",
    "guardrail_intervened",
    "max_tokens",
    "stop_sequence",
    "tool_use",
]
"""Reason for the model ending its response generation.

- "content_filtered": Content was filtered due to policy violation
- "end_turn": Normal completion of the response
- "guardrail_intervened": Guardrail system intervened
- "max_tokens": Maximum token limit reached
- "stop_sequence": Stop sequence encountered
- "tool_use": Model requested to use a tool
"""

_sentinal_value = 34235234523423

class TypedEvent(dict):
    def __post_init__(self):
        if hasattr(self, "as_callback"):
            props = self.as_callback()
            self.update(**props)

    def supports_callback(self):
        return hasattr(self, "as_callback")

    def _invoke_callback(self, callback_handler: Callable, invocation_state = {}) -> None:
        args = self.get_callback_args(invocation_state)
        if args:
            callback_handler(**args)

    def get_callback_args(self, invocation_state = {}):
        if hasattr(self, "as_callback"):
            if hasattr(self, "include_state"):
                self.update(**invocation_state)
            args = self.as_callback()
            return args

        return {}

@dataclass
class StartEventLoopEvent(TypedEvent):

    def as_callback(self) -> list[dict[str, Any]]:
        return {"start_event_loop": True}

@dataclass
class InitEventLoopEvent(TypedEvent):

    def include_state(self):
        return True

    def as_callback(self) -> dict[str, Any]:
        return {"init_event_loop": True}

@dataclass
class ForceStopEvent(TypedEvent):

    reason: str | Exception

    def as_callback(self) -> dict[str, Any]:
        return {"force_stop": True, "force_stop_reason": str(self.reason)}

@dataclass
class MessageEvent(TypedEvent):
    message: Message

    def as_callback(self) -> dict[str, Any]:
        return {"message": self.message}

@dataclass
class EventLoopThrottleDelay(TypedEvent):
    delay: int

    def as_callback(self) -> dict[str, Any]:
        return {"event_loop_throttled_delay": self.delay}

@dataclass
class StopEvent(TypedEvent):
    stop_reason: StopReason
    message: Message
    metrics: "EventLoopMetrics"
    request_state: Any

    @property
    def result(self) -> "AgentResult":
        from strands.agent import AgentResult

        return AgentResult(
            stop_reason=self.stop_reason,
            message=self.message,
            metrics=self.metrics,
            state=self.request_state,
        )

    # def as_callback(self) -> dict[str, Any]:
    #     return { "stop": (self.stop_reason, self.message, self.metrics, self.request_state)}

@dataclass
class StartEvent(TypedEvent):



    def as_callback(self) -> dict[str, Any]:
        return {"start": True}

@dataclass
class ToolResultEvent(TypedEvent):
    tool_result: ToolResult

@dataclass
class ToolResultMessageEvent(TypedEvent):
    message: Any

    def as_callback(self) -> dict[str, Any]:
        return {"message": self.message}

@dataclass
class StreamChunkEvent(TypedEvent):
    chunk: Any

    def as_callback(self) -> dict[str, Any]:
        return {"event": self.chunk}

@dataclass
class StreamDeltaEvent(TypedEvent):
    delta_data: dict[str, Any]

    def include_state(self):
        return True

    def as_callback(self) -> dict[str, Any]:
        # TODO  include invocation state in here
        return self.delta_data

    def _invoke_callback(self, callback_handler: Callable, invocation_state = {}) -> None:

        args = self.as_callback()

        if "delta" in self.delta_data:
            all_args = {**invocation_state, **args}
        else:
            all_args = {**args}

        callback_handler(**all_args)

@dataclass
class ResultEvent(TypedEvent):
    result: "AgentResult"

    def as_callback(self) -> dict[str, Any]:
        return {"result": self.result}

@dataclass
class StreamStopEvent(TypedEvent):
    stop_reason: StopReason
    message: Any
    usage: Any
    metrics: Any

    def as_dict(self) -> dict[str, Any]:
        return {"stop": (self.stop_reason, self.message, self.usage, self.metrics)}