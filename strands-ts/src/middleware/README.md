# Middleware Design Decisions

## Hooks fire outside middleware

Hooks always fire outside the middleware chain. This means:

1. **Hooks always fire, even when middleware short-circuits.** If a middleware returns a cached result without calling `next()`, the Before/After hook pair for that stage still fires.

2. **Middleware retries are not observable to hooks.** If a middleware calls `next()` multiple times (e.g., retry-on-throttle), hooks see one invocation — not three. Internal retries are an implementation detail that should be observed using middleware.

### Execution model

```
BeforeHookEvent          ← always fires
  → Input phase          ← transforms context
    → Output phase       ← wraps Wrap, transforms result on the way out
      → Wrap phase       ← full wrap, may retry/short-circuit
        → terminal       ← actual operation
AfterHookEvent           ← always fires (even on short-circuit)
```

### Consequences

- A rate-limiter middleware that short-circuits still triggers Before/After hooks — consumers monitoring call volume see the attempt even though the model was never called.
- A retry middleware that calls `next()` three times produces one hook pair, not three. If consumers need per-attempt observability, they register middleware (not hooks) at the same stage.
- Resume logic (calling `next()` with different args) is invisible to hooks by design. Hooks see the outermost call/result boundary.

## Middleware cannot resume interrupts

`AgentStreamStage` middleware cannot currently resume tool-level interrupts. Interrupt resolution (`_interruptState.resume()`) runs in `stream()`'s outer loop, outside the middleware chain. When a tool-level interrupt fires, `_stream` catches the `InterruptError` internally and returns a normal `AgentResult` with `stopReason: 'interrupt'` — the middleware sees the result but cannot re-enter the stream with interrupt responses.

To programmatically resume interrupts within a single invocation, use `AfterInvocationEvent.resume` in a hook. A future enhancement could add a resume mechanism to `AgentStreamResult` or `AgentStreamContext` so middleware can signal re-entry with interrupt responses.
