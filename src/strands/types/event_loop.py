"""Event loop-related type definitions for the SDK."""
from dataclasses import dataclass
from typing import Literal, Any

from typing_extensions import TypedDict

from .content import Message


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

class TypedEvent(dict):
    pass


@dataclass
class StartEventLoopEvent(TypedEvent):

    def as_callback(self) -> list[dict[str, Any]]:
        return [{"start": True},
        {"start_event_loop": True}]

@dataclass
class InitEventLoopEvent(TypedEvent):

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