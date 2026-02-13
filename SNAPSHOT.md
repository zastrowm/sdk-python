# Design Doc: Low-Level Snapshot API

**Status**: Implemented

**Date**: 2026-01-28

**Issue**: https://github.com/strands-agents/sdk-python/issues/1138

## Goals

- [x] On-demand state capture — Developers can explicitly capture agent state at any point, independent of automatic session management.
- [ ] Foundation for session management — Snapshots provide a primitive that session managers can use internally, enabling alternative persistence strategies beyond incremental message recording.
- [x] Extensibility via hooks — Plugins and hook providers can contribute additional data to snapshots, allowing custom state to be captured and restored alongside core agent state.
- [x] Configurable scope — Developers control which properties are included in a snapshot, enabling use cases that require only a subset of state (e.g., messages without interrupt state).

## Motivation

Today, developers who want to manually snapshot and restore agent state can *almost* do so by saving and loading these properties directly:

- `Agent.messages` — the conversation history
- `Agent.state` — custom application state
- `Agent._interrupt_state` — internal state for interrupt handling
- Conversation manager internal state — state held by the conversation manager (e.g., sliding window position)

However, this approach is fragile: it requires knowledge of internal implementation details, and the set of properties may change between versions. This proposal introduces a stable, convenient API to accomplish the same thing without relying on internals.

**This API does not change agent behavior** — it simply provides a clean way to serialize and restore the existing state that already exists on the agent.

## Context

Developers need a way to preserve and restore the exact state of an agent at a specific point in time. The existing SessionManagement doesn't address this:

- SessionManager works in the background, incrementally recording messages rather than full state. This means it's not possible to restore to arbitrary points in time.
- After a message is saved, there is no way to modify it and have it recorded in session-management, preventing more advance context-management strategies while being able to pause & restore agents.
- There is no way to proactively trigger session-management (e.g., after modifying `agent.messages` or `agent.state` directly)

## Decision

Add a low-level, explicit snapshot API as an alternative to automatic session-management. This enables preserving the exact state of an agent at a specific point and restoring it later — useful for evaluation frameworks, custom session management, and checkpoint/restore workflows.

### API

```python
from strands.agent import Snapshotter

class Snapshot(TypedDict):
    type: str                # the type of data stored (e.g., "agent"). Strands-owned.
    version: str             # version string for forward compatibility. Strands-owned.
    timestamp: str           # ISO 8601 timestamp of when snapshot was taken. Strands-owned.
    data: dict[str, Any]     # opaque; do not modify — format subject to change. Strands-owned.
    app_data: dict[str, Any] # application-owned data to store with the snapshot

class Snapshotter:
    def __init__(
        self,
        include: SnapshotPreset | list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        """Initialize with default include/exclude settings."""
        ...

    def take(
        self,
        app_data: dict[str, Any] | None = None,
        include: SnapshotPreset | list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Snapshot:
        """Capture the current agent state as a snapshot."""
        ...

    async def take_async(
        self,
        app_data: dict[str, Any] | None = None,
        include: SnapshotPreset | list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Snapshot:
        """Capture the current agent state as a snapshot (async)."""
        ...

    def load(self, snapshot: Snapshot) -> None:
        """Restore agent state from a snapshot."""
        ...

class Agent:
    def __init__(
        self,
        ...,
        snapshots: Snapshotter | None = None,
    ):
        ...

    # Access via agent.snapshots
    snapshots: Snapshotter
```

### Available Snapshot Fields

The following fields can be included in a snapshot:

- `messages` — conversation history
- `state` — custom application state (`agent.state`)
- `conversation_manager_state` — internal state of the conversation manager
- `interrupt_state` — state of any active interrupts
- `system_prompt` — the agent's system prompt

### Include/Exclude Parameters

The `include` and `exclude` parameters control which fields are captured:

- **`include="session"`** — Preset that includes `messages`, `state`, `conversation_manager_state`, and `interrupt_state` (excludes `system_prompt`)
- **`include=["field1", "field2"]`** — Explicit list of fields to include
- **`exclude=["field1"]`** — Fields to exclude (applied after include)

Either `include` or `exclude` must be specified (either as defaults on the Snapshotter or per-call).

### Behavior

Snapshots capture **agent state** (data), not **runtime behavior** (code):

- **Agent State** — Data persisted as part of session-management: conversation messages, context, and other JSON-serializable data. This is what snapshots save and restore.
- **Runtime Behavior** — Configuration that defines how the agent operates: model, tools, ConversationManager, etc. These are *not* included in snapshots and must be set separately when creating or restoring an agent.

The intent is that anything stored or restored by session-management would be stored in a snapshot — so this proposal is *not* documenting or changing what is persisted, but rather providing an explicit way to do what session-management does automatically.

### Contract

- **`app_data`** — Application-owned. Strands does not read, modify, or manage this field. Use it to store checkpoint labels, timestamps, or any application-specific data without the need for a separate/standalone object/datastore.
- **`type`, `version`, `timestamp`, and `data`** — Strands-owned. These fields are managed internally and should be treated as opaque. The format of `data` is subject to change; do not modify or depend on its structure.
- **Serialization** — Strands guarantees that all Strands-owned fields will only contain JSON-serializable values.

### Hooks

A `SnapshotCreatedEvent` hook is fired after `take()` completes, allowing hook providers to add custom data to the snapshot's `app_data` field:

```python
class CustomDataHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(SnapshotCreatedEvent, self.on_snapshot_created)

    def on_snapshot_created(self, event: SnapshotCreatedEvent) -> None:
        event.snapshot["app_data"]["custom_key"] = "custom_value"
```

### FileSystemPersister

A `FileSystemPersister` helper class is provided for basic persistence:

```python
from strands.agent.snapshot import FileSystemPersister

# Save a snapshot
persister = FileSystemPersister(path="checkpoints/snapshot.json")
persister.save(agent.snapshots.take())

# Load a snapshot
snapshot = persister.load()
agent.snapshots.load(snapshot)
```

### Future Concerns

- Snapshotting for MultiAgent constructs - Snapshot is designed in a way that the snapshot could be reused for multi-agent with a similar api
- Providing additional storage APIs for snapshot CRUD operations (database, S3, etc.)
- Providing APIs to customize serialization formats
- Async support for SnapshotCreatedEvent hook

## Developer Experience

### Basic Usage with Defaults

```python
from strands import Agent
from strands.agent import Snapshotter

# Configure defaults once
snapshotter = Snapshotter(include="session")
agent = Agent(tools=[tool1, tool2], snapshots=snapshotter)

# Take snapshots without repeating configuration
snapshot = agent.snapshots.take()
snapshot = agent.snapshots.take(app_data={"checkpoint": "before_update"})

# Restore
agent.snapshots.load(snapshot)
```

### Evaluations via Rewind and Replay

```python
snapshotter = Snapshotter(include="session")
agent = Agent(tools=[tool1, tool2], snapshots=snapshotter)
snapshot = agent.snapshots.take()

result1 = agent("What is the weather?")

# ...

agent2 = Agent(tools=[tool3, tool4], snapshots=snapshotter)
agent2.snapshots.load(snapshot)
result2 = agent2("What is the weather?")
# ... 
# Human/manual evaluation if one outcome was better than the other
# ...
```

### Advanced Context Management

```python
snapshotter = Snapshotter(include="session")
agent = Agent(
    conversation_manager=CompactingConversationManager(),
    snapshots=snapshotter,
)
snapshot = agent.snapshots.take(app_data={"checkpoint": "before_long_task"})

# ... later ...
later_agent = Agent(
    conversation_manager=CompactingConversationManager(),
    snapshots=snapshotter,
)
later_agent.snapshots.load(snapshot)
```

### Persisting Snapshots

```python
from strands.agent.snapshot import FileSystemPersister, Snapshotter

snapshotter = Snapshotter(include="session")
agent = Agent(tools=[tool1, tool2], snapshots=snapshotter)
agent("Remember that my favorite color is orange.")

# Save to file
snapshot = agent.snapshots.take(app_data={"user_id": "123"})
FileSystemPersister("snapshot.json").save(snapshot)

# Later, restore from file
snapshot = FileSystemPersister("snapshot.json").load()

agent = Agent(tools=[tool1, tool2], snapshots=snapshotter)
agent.snapshots.load(snapshot)
agent("What is my favorite color?")  # "Your favorite color is orange."
```

### Overriding Defaults Per-Call

```python
snapshotter = Snapshotter(include="session")
agent = Agent(snapshots=snapshotter)

# Use defaults
snapshot = agent.snapshots.take()

# Override for this call only
snapshot = agent.snapshots.take(include=["messages", "state"])
snapshot = agent.snapshots.take(exclude=["interrupt_state"])
```

### Edge cases

Restoring runtime behavior (e.g., tools) is explicitly not supported:

```python
snapshotter = Snapshotter(include="session")
agent1 = Agent(tools=[tool1, tool2], snapshots=snapshotter)
snapshot = agent1.snapshots.take()
agent_no = Agent(snapshot)  # tools are NOT restored
```

## State Boundary

The implementation draws a distinction between "evolving state" (data that changes as the agent runs) and "agent definition" (configuration that defines what the agent *is*):

| Evolving State (snapshotted) | Agent Definition (not snapshotted) |
|------------------------------|-----------------------------------|
| messages | tools |
| state | model |
| conversation_manager_state | conversation_manager |
| interrupt_state | callback_handler |
| system_prompt (optional) | hooks |

The `system_prompt` field is available but excluded from the "session" preset since it's typically considered part of agent definition rather than evolving state.

## Consequences

**Easier:**
- Building evaluation frameworks with rewind/replay capabilities
- Implementing custom session management strategies
- Creating checkpoints during long-running agent tasks
- Cloning agents (load the same snapshot into multiple agent instances)
- Resetting agents to a known state (we do this manually for Graphs)

**More difficult:**
- N/A — this is an additive API

## Willingness to Implement

Yes — Implemented.
