# Middleware Test Coverage Gaps

Compared against the TypeScript implementation (PR #2681). Gaps are grouped by area.
Items marked "expected gap" are for features not yet ported (AgentStreamStage, custom stages).

---

## Registry-level gaps

### 1. Chained context modification across multiple Wrap handlers

Multiple Wrap handlers each modify context — terminal sees accumulated modifications.

```python
# MW A: adds "a"=True
# MW B: adds "b"=True  
# Terminal should see both "a" and "b"
```

### 2. Middleware error transformation

Middleware catches an error from next and re-throws a different one:

```python
async def transformer(context, next_fn):
    try:
        async for event in next_fn(context):
            yield event
    except ValueError as e:
        raise RuntimeError(f"Wrapped: {e}") from e
# Caller sees RuntimeError, not ValueError
```

### 3. CancelledError / InterruptException propagation

Verify these special errors propagate through middleware without being swallowed
by broad `except Exception` patterns in middleware layers.

### 4. Multi-layer finally ordering

When terminal throws, ALL finally blocks in ALL middleware layers run, in
reverse order (inner-to-outer):

```python
# outer: try/finally → "outer_finally"
# inner: try/finally → "inner_finally"
# terminal: raises
# Expected order: ["inner_finally", "outer_finally"]
```

### 5. Two-layer cleanup on generator abandon (aclose)

When consumer calls `aclose()`, both middleware layers' finally blocks run.
Currently only tested with single layer.

### 6. Remove only first occurrence of duplicate handler

```python
registry.add(stage, handler)
registry.add(stage, handler)
registry.remove(stage, handler)
# handler still runs once (second registration remains)
```

### 7. Remove no-op for stage never used

Call `remove()` on a stage that has never had any handlers registered.

---

## InvokeModelStage gaps

### 8. Short-circuit verifies model NOT called

Spy on `model.stream` to confirm it was never invoked when middleware
short-circuits.

### 9. Context transform: modified messages reach the model

Spy to verify that when middleware transforms `context.messages`, the model
actually receives the modified messages in `stream_messages()`.

### 10. Hooks fire even when middleware short-circuits

If middleware never calls `next`, AfterModelCallEvent should still fire
with the short-circuited result.

### 11. Cleanup only removes the specific handler, not siblings

Register two handlers, remove one — verify the other still fires.

### 12. Phase ordering tested at agent level

Register Output first, Input second, Wrap third at the agent level — verify
Input still runs first. Catches bugs where `add_middleware` doesn't delegate
phases correctly.

### 13. Cleanup removes Input/Output handlers at agent level

Verify cleanup functions returned from `.Input` and `.Output` registrations
work correctly via `agent.add_middleware(Stage.Input, ...)`.

### 14. Unknown phase throws error

Exercise the `raise ValueError` for unknown `_phase` values.

### 15. No-middleware baseline tests

Explicit tests that agent without middleware works correctly for text response,
tool use response, and streaming — regression guard.

### 16. Auto-retry on throttle use case

Full agent-level integration test: middleware catches ThrottlingException,
retries by calling next again.

---

## ExecuteToolStage gaps

### 17. Short-circuit verifies real tool NOT called

When middleware returns cached result, verify the tool function itself was
never invoked (spy/flag on the tool).

### 18. Context transform: modified input reaches tool function

When middleware transforms `context.tool_use["input"]`, the tool function
receives the modified input arguments.

### 19. AfterToolCallEvent fires after middleware

Explicitly verify AfterToolCallEvent fires after middleware completes and
receives the middleware's result.

### 20. Hooks fire when middleware short-circuits

If middleware never calls next, AfterToolCallEvent should still fire with
the short-circuited result.

### 21. AfterToolCallEvent receives middleware result on short-circuit

When middleware provides a mock result without calling next, does
`AfterToolCallEvent.result` contain that mock result?

### 22. Caching plugin use case

Full ToolResultCache plugin: first call executes tool, second call with
same input returns cached result without executing tool again.

---

## Interrupt gaps

### 23. InterruptEvent visible on the stream

When middleware calls `context.interrupt()`, verify the resulting
`ToolInterruptEvent` appears in the stream (observable by callback handlers).

### 24. Context replace preserves interrupt function

`dataclasses.replace(context, tool_use=modified)` should preserve
`_interrupt_state` and keep `interrupt()` working.

---

## Expected gaps (not yet implemented)

- **AgentStreamStage** — wraps full agent stream, interrupt support, event filtering/injection
- **Custom stages** — `createStage` equivalent, reference identity, third-party usage
