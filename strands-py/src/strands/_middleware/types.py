"""Middleware type system for the Strands Agents framework.

Defines stage tokens, phase sub-tokens, the internal result sentinel, and handler type aliases.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar, Union

TContext = TypeVar("TContext")
TResult = TypeVar("TResult")
TEvent = TypeVar("TEvent")


@dataclass
class MiddlewareResult(Generic[TResult]):
    """Internal sentinel yielded as the last item to communicate the result up the chain.

    Not part of the public API. Integration sites strip this from the event stream.
    """

    value: TResult


class MiddlewareInputPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Input handlers — transforms context before execution."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        self._stage = stage
        self._phase = "input"

    def __repr__(self) -> str:
        return f"MiddlewareInputPhase(stage={self._stage.name!r})"


class MiddlewareWrapPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Wrap handlers — full async generator wrap."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        self._stage = stage
        self._phase = "wrap"

    def __repr__(self) -> str:
        return f"MiddlewareWrapPhase(stage={self._stage.name!r})"


class MiddlewareOutputPhase(Generic[TContext, TResult, TEvent]):
    """Phase sub-token for Output handlers — transforms result after execution."""

    __slots__ = ("_stage", "_phase")

    def __init__(self, stage: MiddlewareStage[TContext, TResult, TEvent]) -> None:
        self._stage = stage
        self._phase = "output"

    def __repr__(self) -> str:
        return f"MiddlewareOutputPhase(stage={self._stage.name!r})"


class MiddlewareStage(Generic[TContext, TResult, TEvent]):
    """A stage token identifying a middleware interception point.

    Stages are module-level singletons used as registry keys. Each stage carries
    three phase sub-tokens: `.Input`, `.Wrap`, `.Output`.
    """

    __slots__ = ("name", "Input", "Wrap", "Output")

    def __init__(self, name: str) -> None:
        self.name = name
        self.Input: MiddlewareInputPhase[TContext, TResult, TEvent] = MiddlewareInputPhase(self)
        self.Wrap: MiddlewareWrapPhase[TContext, TResult, TEvent] = MiddlewareWrapPhase(self)
        self.Output: MiddlewareOutputPhase[TContext, TResult, TEvent] = MiddlewareOutputPhase(self)

    def __repr__(self) -> str:
        return f"MiddlewareStage(name={self.name!r})"

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other


MiddlewareNext = Callable[[Any], AsyncGenerator[Any, None]]
"""Type alias for the `next` function passed to Wrap handlers.

Signature: (context) -> AsyncGenerator[event | MiddlewareResult, None]
"""

MiddlewareHandler = Callable[[Any, MiddlewareNext], AsyncGenerator[Any, None]]
"""Type alias for Wrap phase handlers.

Signature: async def handler(context, next_fn) -> AsyncGenerator[event | MiddlewareResult, None]
"""

MiddlewareInputHandler = Callable[[Any], Union[Any, Awaitable[Any]]]
"""Type alias for Input phase handlers.

Signature: def handler(context) -> context  (sync or async)
"""

MiddlewareOutputHandler = Callable[[Any], Union[Any, Awaitable[Any]]]
"""Type alias for Output phase handlers.

Signature: def handler(result) -> result  (sync or async)
"""


def _is_awaitable(obj: Any) -> bool:
    """Check if an object is awaitable (coroutine or has __await__)."""
    return inspect.isawaitable(obj)
