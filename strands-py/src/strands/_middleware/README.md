# Python Middleware — Differences from TypeScript

This documents intentional differences between the Python and TypeScript middleware implementations.

## Result encoding

TypeScript async generators can `return value`, captured by `yield*`. Python async generators cannot return values. The Python implementation uses a `_MiddlewareResult` sentinel — the terminal and each middleware layer yield it as the final item in the generator. Integration sites strip it from the event stream.

**Wrap handler pattern:**
```python
async def my_middleware(context, next_fn):
    async for event in next_fn(context):
        if isinstance(event, _MiddlewareResult):
            yield _MiddlewareResult(transform(event.value))
        else:
            yield event
```

Pass-through is simpler — just forward everything:
```python
async def passthrough(context, next_fn):
    async for event in next_fn(context):
        yield event
```

## Interrupt source ID format

TypeScript uses `{idPrefix}:{params.name}` as the interrupt ID (plain name string):
```
middleware:executeTool:{toolUseId}:{name}
```

Python uses `uuid5` to hash the name, matching the existing Python hook interrupt pattern (`_Interruptible._interrupt_id`):
```
middleware:executeTool:{toolUseId}:{uuid5(NAMESPACE_OID, name)}
```

This means interrupt IDs are **not portable** between the TypeScript and Python SDKs. This is acceptable because interrupt state is per-session and per-language.

## Private module

The middleware module is `_middleware/` (underscore-prefixed) — not publicly importable. Public access is exclusively via `strands.__init__` exports (`InvokeModelStage`, `ExecuteToolStage`, etc.). TypeScript exports from `@strands-agents/sdk` directly.

## Stages implemented

| Stage | TypeScript | Python |
|-------|:----------:|:------:|
| `InvokeModelStage` | Yes | Yes |
| `ExecuteToolStage` | Yes | Yes |
| `AgentStreamStage` | Yes | Not yet |

## `createStage` not exposed

Same as TypeScript — `createStage` is internal. Only the built-in stages are public.

## Context is a dataclass

TypeScript uses plain interfaces for contexts. Python uses `@dataclass` which enables `dataclasses.replace()` for immutable-style context transformation:
```python
from dataclasses import replace
modified = replace(context, system_prompt="Injected")
```

## Handler signature

TypeScript Wrap handlers are `async function*(context, next)` returning the result via generator `return`.

Python Wrap handlers are `async def handler(context, next_fn)` returning an `AsyncGenerator`. The result is communicated by yielding `_MiddlewareResult(value)` as the last item.

Input/Output phase handlers are identical in spirit: plain `(context) -> context` or `(result) -> result` (sync or async).
