"""Middleware registry for composing handler chains.

The registry stores handlers keyed by stage tokens and composes them into
execution chains with fixed phase ordering.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

from .types import (
    MiddlewareHandler,
    MiddlewareInputHandler,
    MiddlewareInputPhase,
    MiddlewareNext,
    MiddlewareOutputHandler,
    MiddlewareOutputPhase,
    MiddlewareStage,
    _MiddlewareResult,
)

# Compose layering: input (outermost) → output → wrap (innermost, closest to terminal).
# Execution order: input → wrap → output.
_PHASE_ORDER: dict[str, int] = {"input": 0, "output": 1, "wrap": 2}


@dataclass
class _TaggedHandler:
    """Internal entry pairing a handler with its phase for ordering."""

    phase: str
    handler: MiddlewareHandler


class MiddlewareRegistry:
    """Registry that stores middleware handlers keyed by stage tokens and composes them into chains."""

    def __init__(self) -> None:
        self._handlers: dict[MiddlewareStage[Any, Any, Any], list[_TaggedHandler]] = {}

    def add(self, stage: MiddlewareStage[Any, Any, Any], handler: MiddlewareHandler) -> None:
        """Register a Wrap phase handler for the given stage.

        Handlers are stored in registration order within their phase.
        """
        handlers = self._handlers.setdefault(stage, [])
        handlers.append(_TaggedHandler(phase="wrap", handler=handler))

    def add_input(
        self, phase: MiddlewareInputPhase[Any, Any, Any], handler: MiddlewareInputHandler
    ) -> MiddlewareHandler:
        """Register an Input phase handler. Returns the adapted handler for removal."""
        stage = phase._stage

        async def adapted(context: Any, next_fn: MiddlewareNext) -> AsyncGenerator[Any, None]:
            transformed = handler(context)
            if inspect.isawaitable(transformed):
                transformed = await transformed
            async for event in next_fn(transformed):
                yield event

        handlers = self._handlers.setdefault(stage, [])
        handlers.append(_TaggedHandler(phase="input", handler=adapted))
        return adapted

    def add_output(
        self, phase: MiddlewareOutputPhase[Any, Any, Any], handler: MiddlewareOutputHandler
    ) -> MiddlewareHandler:
        """Register an Output phase handler. Returns the adapted handler for removal."""
        stage = phase._stage

        async def adapted(context: Any, next_fn: MiddlewareNext) -> AsyncGenerator[Any, None]:
            async for event in next_fn(context):
                if isinstance(event, _MiddlewareResult):
                    transformed = handler(event.value)
                    if inspect.isawaitable(transformed):
                        transformed = await transformed
                    yield _MiddlewareResult(transformed)
                else:
                    yield event

        handlers = self._handlers.setdefault(stage, [])
        handlers.append(_TaggedHandler(phase="output", handler=adapted))
        return adapted

    def remove(self, stage: MiddlewareStage[Any, Any, Any], handler: MiddlewareHandler) -> None:
        """Remove the first occurrence of a handler from a stage (by reference equality)."""
        handlers = self._handlers.get(stage)
        if not handlers:
            return
        for i, tagged in enumerate(handlers):
            if tagged.handler is handler:
                handlers.pop(i)
                return

    def compose(self, stage: MiddlewareStage[Any, Any, Any], terminal: MiddlewareNext) -> MiddlewareNext:
        """Compose all registered handlers for a stage into a single chain.

        Handlers are ordered by phase (input → output → wrap), then by registration order
        within each phase. First in the composed chain = outermost.

        Returns the terminal directly if no handlers are registered (zero overhead fast path).
        """
        tagged = self._handlers.get(stage)
        if not tagged:
            return terminal

        # Stable sort by phase order
        sorted_handlers = sorted(tagged, key=lambda t: _PHASE_ORDER[t.phase])

        # Build chain from inside-out (rightmost handler wraps terminal first)
        current: MiddlewareNext = terminal
        for i in range(len(sorted_handlers) - 1, -1, -1):
            handler = sorted_handlers[i].handler
            next_fn = current

            def _make_layer(h: MiddlewareHandler, nf: MiddlewareNext) -> MiddlewareNext:
                async def layer(ctx: Any) -> AsyncGenerator[Any, None]:
                    inner_gens: list[AsyncGenerator[Any, None]] = []

                    def tracking_next(c: Any) -> AsyncGenerator[Any, None]:
                        gen = nf(c)
                        inner_gens.append(gen)
                        return gen

                    handler_gen = h(ctx, tracking_next)
                    try:
                        async for event in handler_gen:
                            yield event
                    finally:
                        await handler_gen.aclose()
                        for gen in inner_gens:
                            await gen.aclose()

                return layer

            current = _make_layer(handler, next_fn)

        return current

    async def invoke(
        self,
        stage: MiddlewareStage[Any, Any, Any],
        context: Any,
        terminal: MiddlewareNext,
    ) -> AsyncGenerator[Any, None]:
        """Compose and invoke the middleware chain for a stage.

        Yields all events from the chain including the _MiddlewareResult sentinel.
        The caller is responsible for stripping _MiddlewareResult from the stream.
        """
        chain = self.compose(stage, terminal)
        gen = chain(context)
        try:
            async for event in gen:
                yield event
        finally:
            await gen.aclose()
