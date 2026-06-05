# Middleware API Bar-Raising

## Motivation

Hooks let you observe operations and set flags, but they don't let you wrap them. If you want to do something both before and after a model call — time it, add a span, catch errors — hooks force you to manage state across two separate callbacks:

```typescript
// With hooks: state management across two disconnected callbacks
let startTime: number
agent.addHook(BeforeModelCallEvent, () => { startTime = Date.now() })
agent.addHook(AfterModelCallEvent, () => { metrics.record(Date.now() - startTime) })
// With middleware: one function, natural scoping
agent.addMiddleware(InvokeModelStage, async function* (context, next) {
  const start = Date.now()
  const result = yield* next(context)
  metrics.record(Date.now() - start)
  return result
})
```

Beyond before/after, middleware makes caching, input transformation, short-circuiting, and error handling natural to express. All of these are awkward or impossible with hooks alone.

## Use Cases

1. **Wrap before/after** — timing, telemetry spans, logging (natural scoping vs two disconnected hook callbacks)
2. **Input transformation** — sanitize messages, inject system prompt fragments, modify tool specs
3. **Input transformation** — stream messages in a different format (like OpenAIResponses)
4. **Short-circuiting** — return mock/cached result without calling next
5. **Error handling** — try/catch around model or tool execution, retry logic
6. **Event filtering/injection** — suppress noisy stream events, inject synthetic ones

### Plugin examples (from tests)

**Tool result cache** — a plugin that caches tool results by name+input, skipping execution on repeat calls:

```typescript
class ToolResultCache implements Plugin {
  name = 'tool-result-cache'
  private readonly _cache = new Map<string, ToolResultBlock>()

  initAgent(agent: LocalAgent): void {
    const cache = this._cache
    agent.addMiddleware(ExecuteToolStage, async function* (context, next) {
      const key = `${context.toolUse.name}:${JSON.stringify(context.toolUse.input)}`
      const cached = cache.get(key)
      if (cached) {
        return { result: new ToolResultBlock({ toolUseId: context.toolUse.toolUseId, status: cached.status, content: cached.content }) }
      }
      const result = yield* next(context)
      cache.set(key, result.result)
      return result
    })
  }
}
```

**Auto-retry on throttle** — a plugin that retries model calls on transient errors:

```typescript
class RetryOnThrottle implements Plugin {
  name = 'retry-on-throttle'
  constructor(private readonly _maxRetries = 3) {}

  initAgent(agent: LocalAgent): void {
    const maxRetries = this._maxRetries
    agent.addMiddleware(InvokeModelStage, async function* (context, next) {
      for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
          return yield* next(context)
        } catch (e) {
          const isRetryable = (e as Error).message.includes('ThrottlingException')
          if (!isRetryable || attempt === maxRetries - 1) throw e
        }
      }
      throw new Error('exhausted retries')
    })
  }
}
```

**Stream final turn only** — buffers content events from intermediate tool-use turns, only emitting the final response:

```typescript
class StreamFinalTurnOnly implements Plugin {
  name = 'stream-final-turn-only'

  initAgent(agent: LocalAgent): void {
    agent.addMiddleware(AgentStreamStage, (...args) => this._handler(...args))
  }

  private async *_handler(
    ...[context, next]: Parameters<MiddlewareHandlerOf<typeof AgentStreamStage>>
  ): ReturnType<MiddlewareHandlerOf<typeof AgentStreamStage>> {
    let buffer: AgentStreamEvent[] = []
    const gen = next(context)
    let iterResult = await gen.next()
    while (!iterResult.done) {
      const event = iterResult.value
      if (event.type === 'contentBlockEvent' || event.type === 'modelStreamUpdateEvent') {
        buffer.push(event)
      } else if (event.type === 'afterModelCallEvent') {
        const stopReason = (event as AfterModelCallEvent).stopData?.stopReason
        if (stopReason === 'endTurn') {
          for (const buffered of buffer) yield buffered
        }
        buffer = []
        yield event
      } else {
        yield event
      }
      iterResult = await gen.next()
    }
    for (const buffered of buffer) yield buffered
    return iterResult.value
  }
}
```

## Public API Surface

### Registration

```typescript
import { Agent, InvokeModelStage, ExecuteToolStage, AgentStreamStage, createStage } from '@strands-agents/sdk'
import type { MiddlewareStage, MiddlewareHandler, MiddlewareNext, MiddlewareHandlerOf, MiddlewareNextOf } from '@strands-agents/sdk'

agent.addMiddleware(InvokeModelStage, async function* (context, next) {
  const result = yield* next(context)
  return result
})
```

`addMiddleware` returns a cleanup function (consistent with `addHook`) for middleware removal.

### Built-in stages

| Stage | Wraps | Context fields |
|-------|-------|----------------|
| `InvokeModelStage` | Model call (between Before/AfterModelCallEvent) | messages (readonly), systemPrompt, toolSpecs (readonly), toolChoice, modelState, invocationState |
| `ExecuteToolStage` | Single tool execution (between Before/AfterToolCallEvent) | tool, toolUse (name, id, input), invocationState, interrupt() |
| `AgentStreamStage` | Full `agent.stream()` output | args, options, interrupt() |

### Middleware interrupts

```typescript
agent.addMiddleware(ExecuteToolStage, async function* (context, next) {
  const { response } = context.interrupt<string>({ name: 'approve', reason: 'Confirm?' })
  if (response !== 'yes') { /* short-circuit */ }
  return yield* next(context)
})
```

Returns `MiddlewareInterruptResult<T>` (wrapper object) — intentionally different from tool/hook `Interruptible` which returns `T` directly. The wrapper allows non-breaking additions (cached data, metadata) in the future.

### Utility types

```typescript
// Extract handler type from a stage token
type MyHandler = MiddlewareHandlerOf<typeof InvokeModelStage>

// Extract next-function type from a stage token  
type MyNext = MiddlewareNextOf<typeof AgentStreamStage>
```

### Custom stages (third-party)

```typescript
interface MyContext { agent: LocalAgent; data: string }
interface MyResult { output: string }

const MyStage = createStage<MyContext, AgentStreamEvent, MyResult>('myCustomStage')
```

Third parties can define stages without modifying SDK internals. Stage tokens are frozen objects keyed by reference — two stages with the same name string are still distinct.

### Module exports

All exported from `@strands-agents/sdk` (flat namespace per decision record):

**Values:** `createStage`, `InvokeModelStage`, `ExecuteToolStage`, `AgentStreamStage`

**Types:** `MiddlewareStage`, `MiddlewareHandler`, `MiddlewareNext`, `MiddlewareHandlerOf`, `MiddlewareNextOf`, `InvokeModelContext`, `ExecuteToolContext`, `AgentStreamContext`, `MiddlewareInterruptResult`, `MiddlewareInterruptible`

## Key Design Decisions

| Decision | Rationale | Tenet |
|----------|-----------|-------|
| Async generators (not plain async) | Operations stream events; plain async would buffer and kill latency | Simple at any scale |
| First registered = outermost | Matches Express/Koa convention developers already know | Embrace common standards |
| `addMiddleware` returns cleanup function | Matches `addHook` API; supports dynamic/per-session middleware | Composability |
| `MiddlewareInterruptResult<T>` wrapper | Future-proof; adding fields later is non-breaking | Pay for play |
| `readonly` arrays on `InvokeModelContext` | Enforce immutable-context pattern at type level | The obvious path is the happy path |
| Middleware-prefixed type names | Avoids ambiguity with `Stage` (too generic), `HandlerOf`/`NextOf` (no domain context) | Avoid overloading domain terms |
| Hooks fire unconditionally around middleware | Observability stays intact even when middleware short-circuits | Simple at any scale |

## Plugins as the Primary Registration Path

Middleware is designed to be registered through Plugins. Direct `agent.addMiddleware()` calls are available for quick prototyping and tests, but production middleware should live in a Plugin's `initAgent`:

```typescript
class RateLimiter implements Plugin {
  name = 'rate-limiter'

  initAgent(agent: LocalAgent): void {
    agent.addMiddleware(InvokeModelStage, async function* (context, next) {
      await this.acquireToken()
      return yield* next(context)
    })
  }
}

const agent = new Agent({ plugins: [new RateLimiter()] })
```

This aligns with existing SDK patterns — `RetryStrategy`, `ConversationManager`, and `ModelPlugin` all register behavior through the Plugin interface. Middleware is the mechanism; Plugins are how you package and distribute it.

A Plugin can register middleware for multiple stages, combine middleware with hooks (e.g., middleware for flow control + hooks for metrics emission), and manage its own lifecycle (private state, cleanup via the returned function).

## Relationship to Hooks

**Hooks are for observing.** They fire unconditionally and let consumers react to lifecycle events (log messages added, trace model calls, monitor context changes). They cannot alter control flow.

**Middleware is for controlling flow and changing how the agent behaves.** It wraps operations with full control flow: before/after, try/catch, retry, short-circuit, input transformation. Middleware determines *what happens*; hooks observe *that it happened*.

In a future v2, we would expect to pare hooks down to those needed purely for observation (MessageAdded, ContextChanged, etc.) and move all flow-altering behavior (cancel, retry, resume) into middleware where it composes naturally.

## Not Addressed (intentionally deferred)

- **Telemetry span asymmetry**
- **`interrupt()` data/cache field** — `MiddlewareInterruptResult` wrapper exists to support attaching opaque data to interrupts (e.g., cached conversation state for resumption) without a breaking change. The use case: a middleware interrupts to ask for approval, and wants to store the messages-so-far so that on resume it can skip re-processing. Today you'd have to stash that in external state; with a `data` field on the interrupt, it round-trips automatically.

---

## Appendix: Type Definitions

```typescript
// --- Core types (middleware/types.ts) ---

interface MiddlewareStage<TContext, TEvent, TResult> {
  readonly name: string
  readonly _types?: { context: TContext; event: TEvent; result: TResult }
}

type MiddlewareNext<TContext, TEvent, TResult> = (
  context: TContext
) => AsyncGenerator<TEvent, TResult, undefined>

type MiddlewareHandler<TContext, TEvent, TResult> = (
  context: TContext,
  next: MiddlewareNext<TContext, TEvent, TResult>
) => AsyncGenerator<TEvent, TResult, undefined>

type MiddlewareHandlerOf<S> =
  S extends MiddlewareStage<infer C, infer E, infer R> ? MiddlewareHandler<C, E, R> : never

type MiddlewareNextOf<S> =
  S extends MiddlewareStage<infer C, infer E, infer R> ? MiddlewareNext<C, E, R> : never

// --- Stage factory (middleware/stages.ts) ---

function createStage<TContext, TEvent, TResult>(name: string): MiddlewareStage<TContext, TEvent, TResult>

// --- Interrupt types (middleware/stages.ts) ---

interface MiddlewareInterruptResult<T = JSONValue> {
  response: T
}

interface MiddlewareInterruptible {
  interrupt<T = JSONValue>(params: InterruptParams): MiddlewareInterruptResult<T>
}

// --- Stage contexts ---

interface InvokeModelContext {
  readonly agent: LocalAgent
  readonly messages: readonly Message[]
  readonly systemPrompt?: SystemPrompt
  readonly toolSpecs: readonly ToolSpec[]
  readonly toolChoice?: ToolChoice
  readonly modelState: StateStore
  readonly invocationState: InvocationState
}

interface ExecuteToolContext extends MiddlewareInterruptible {
  readonly agent: LocalAgent
  readonly tool: Tool | undefined
  readonly toolUse: ToolUseData
  readonly invocationState: InvocationState
}

interface AgentStreamContext extends MiddlewareInterruptible {
  readonly agent: LocalAgent
  readonly args: InvokeArgs
  readonly options?: InvokeOptions
}

// --- Built-in stage tokens (return types are the domain objects directly) ---

const InvokeModelStage: MiddlewareStage<InvokeModelContext, AgentStreamEvent, StreamAggregatedResult>
const ExecuteToolStage: MiddlewareStage<ExecuteToolContext, AgentStreamEvent, ToolResultBlock>
const AgentStreamStage: MiddlewareStage<AgentStreamContext, AgentStreamEvent, AgentResult>

// --- Agent method ---

class Agent {
  addMiddleware<TContext, TEvent, TResult>(
    stage: MiddlewareStage<TContext, TEvent, TResult>,
    handler: MiddlewareHandler<TContext, TEvent, TResult>
  ): () => void
}
```

