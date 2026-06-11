# Python vs TypeScript Middleware — Implementation Comparison

## 1. Missing Features (not yet implemented)

### AgentStreamStage

TS wraps the entire agent output stream (`_streamWithMiddleware`). Middleware can filter/inject/buffer events across the full invocation. Python does not have this stage.

### Utility type aliases

TS provides `MiddlewareHandlerOf<S>` and `MiddlewareNextOf<S>` to extract handler types from a stage token — useful for typing class properties that hold handler references. Python has no equivalent.

### Generic `MiddlewareInterruptResult<T>`

TS's interrupt result is `MiddlewareInterruptResult<T = JSONValue>` allowing callers to assert the response shape. Python's is `MiddlewareInterruptResult` with `response: Any`.

---

## 2. Structural Differences (potentially worth aligning)

### `ExecuteToolResult` field naming

- **TS**: `{ result: ToolResultBlock }`
- **Python**: `ExecuteToolResult(tool_result: dict, exception: Exception | None)`

The field is named `result` in TS vs `tool_result` in Python. Python also carries `exception` which TS does not. The `exception` field is needed because the Python tool execution flow captures exceptions into error results (rather than propagating them), and middleware/hooks need access to the original exception for logging/retry decisions.

### `InvokeModelResult` shape

- **TS**: `{ result: StreamAggregatedResult }` — single wrapped field
- **Python**: `InvokeModelResult(stop_reason, message, usage, metrics)` — flattened fields

TS wraps because it wanted a single-field wrapper for future extension. Python flattens because the dataclass itself is the extension point — new fields can be added with defaults. Both are non-breakingly extensible. The flat shape is more ergonomic for Python middleware authors who want to inspect/transform specific fields via `dataclasses.replace()`.

### Interrupt source field

- **TS**: `Interrupt` has `source: 'middleware'` set when raised from middleware
- **Python**: `Interrupt` dataclass has no `source` field

Known gap — documented in README.

---

## 3. Intentional Divergences (documented, by design)

### Result encoding

TS uses async generator `return` values propagated via `yield*`. Python uses `_MiddlewareResult` sentinel yielded as the last event (Python async generators cannot `return` values).

### No removal / cleanup

TS `addMiddleware` returns a cleanup function. Python `add_middleware` returns `None`. Matches the Python hook system which also has no removal.

### Interrupt ID format

- **TS**: `middleware:executeTool:{toolUseId}:{name}` (raw name)
- **Python**: `middleware:executeTool:{toolUseId}:{uuid5(NAMESPACE_OID, name)}` (hashed)

Python uses UUID5 to match the existing Python hook interrupt pattern (`_Interruptible._interrupt_id`). IDs are not portable between SDKs.

### Generator cleanup

- **TS**: `yield*` delegation automatically calls `.return()` on the inner generator when the outer generator completes or is returned early. This triggers `finally` blocks in middleware.
- **Python**: `invoke()` uses `try/finally` with explicit `aclose()` to ensure middleware generators are cleaned up if iteration is abandoned early.

Both correctly clean up generators — the mechanisms differ due to language semantics (`yield*` vs `async for`).

### Single unified dispatch

TS has separate `add()`, `addInput()`, `addOutput()` methods on the registry (plus overloaded `addMiddleware` on Agent). Python has a single `add_middleware` on both registry and agent that dispatches by token type.

### `system_prompt` + `system_prompt_content` in `InvokeModelContext`

Python carries both `system_prompt: str | None` and `system_prompt_content: list | None`. This reflects the Python SDK's dual system-prompt representation used throughout (`Model.stream()`, `stream_messages()`, `Agent.system_prompt`). The string form is for backwards compatibility; content blocks support features like cache points. TS unifies these into a single `systemPrompt?: SystemPrompt` type.

Both are exposed on the middleware context so middleware can transform them independently (e.g., inject text into the string form while preserving cache point blocks).

---

## 4. Potential Issues

### Context immutability

- **TS**: `readonly` fields and `readonly` arrays prevent accidental mutation at compile time
- **Python**: Mutable dataclasses allow `context.messages.append(...)` silently

The documented pattern is `dataclasses.replace()` for transformations, but nothing enforces it. Consider `frozen=True` on context dataclasses in the future (would require removing the `_interrupt_state` mutable field or using `object.__setattr__` for it).

### No fast-path in `invoke()`

`compose()` has a fast path returning the terminal directly when no handlers are registered. But `invoke()` always wraps it in its own async generator function regardless. Adds one generator frame of overhead even with zero middleware. Could be optimized by checking `compose() is terminal` and delegating directly.
