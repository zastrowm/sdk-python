"""Middleware system for wrapping agent stages.

Middleware controls flow (retry, cache, transform, short-circuit) around
agent operations using async generator handlers.
"""

from .registry import MiddlewareRegistry
from .stages import InvokeModelContext, InvokeModelResult, InvokeModelStage
from .types import (
    MiddlewareHandler,
    MiddlewareInputHandler,
    MiddlewareInputPhase,
    MiddlewareNext,
    MiddlewareOutputHandler,
    MiddlewareOutputPhase,
    MiddlewareStage,
    MiddlewareWrapPhase,
    _MiddlewareResult,
)

__all__ = [
    "InvokeModelContext",
    "InvokeModelResult",
    "InvokeModelStage",
    "MiddlewareHandler",
    "MiddlewareInputHandler",
    "MiddlewareInputPhase",
    "MiddlewareNext",
    "MiddlewareOutputHandler",
    "MiddlewareOutputPhase",
    "MiddlewareRegistry",
    "MiddlewareStage",
    "MiddlewareWrapPhase",
    "_MiddlewareResult",
]
