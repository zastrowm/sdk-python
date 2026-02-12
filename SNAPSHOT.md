# Design Doc: Low-Level Snapshot API

**Status**: Implemented

**Date**: 2026-01-28

**Issue**: https://github.com/strands-agents/sdk-python/issues/1138

## Motivation

Today, developers who want to manually snapshot and restore agent state can *almost* do so by saving and loading these properties directly:

- `Agent.messages` — the conversation history
- `Agent.state` — custom application state
- `Agent._interrupt_state` — internal state for interrupt handling
- Conversation manager internal state — state held by the conversation manager (e.g., sliding window position)

However, this approach is fragile: it requires knowledge of internal implementation details, and the set of properties may change between versions. This proposal introduces a stable, convenient API to accomplish the same thing without relying on internals.

**This API does not change agent behavior** — it simply provides a clean way to serialize and restore the existing state that already exists on the agent.

## Goals

- [ ] On-demand state capture — Developers can explicitly capture agent state at any point, independent of automatic session management.
- [ ] Foundation for session management — Snapshots provide a primitive that session managers can use internally, enabling alternative persistence strategies beyond incremental message recording.
- [ ] Extensibility via hooks — Plugins and hook providers can contribute additional data to snapshots, allowing custom state to be captured and restored alongside core agent state.
- [ ] Configurable scope — Developers control which properties are included in a snapshot, enabling use cases that require only a subset of state (e.g., messages without interrupt state).

## Context

Developers need a way to preserve and restore the exact state of an agent at a specific point in time. The existing SessionManagement doesn't address this:

- SessionManager works in the background, incrementally recording messages rather than full state. This means it's not possible to restore to arbitrary points in time.
- After a message is saved, there is no way to modify it and have it recorded in session-management, preventing more advance context-management strategies while being able to pause & restore agents.
- There is no way to proactively trigger session-management (e.g., after modifying `agent.messages` or `agent.state` directly)

## Decision

Add a low-level, explicit snapshot API as an alternative to automatic session-management. This enables preserving the exact state of an agent at a specific point and restoring it later — useful for evaluation frameworks, custom session management, and checkpoint/restore workflows.

### API

```python
from typing import Literal

# Named preset for include parameter
SnapshotPreset = Literal["session"]

class Snapshot(TypedDict):
    type: str                # the type of data stored (e.g., "agent"). Strands-owned.
    version: str             # version string for forward compatibility. Strands-owned.
    timestamp: str           # ISO 8601 timestamp of when snapshot was taken. Strands-owned.
    data: dict[str, Any]     # opaque; do not modify — format subject to change. Strands-owned.
    app_data: dict[str, Any] # application-owned data to store with the snapshot

class Agent:
    def take_snapshot(
        self,
        app_data: dict[str, Any] | None = None,
        include: SnapshotPreset | list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Snapshot:
        """Capture the current agent state as a snapshot."""
        ...

    def load_snapshot(self, snapshot: Snapshot) -> None:
        """Restore agent state from a snapshot."""
        ...
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

Either `include` or `exclude` must be specified.

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

A `SnapshotCreatedEvent` hook is fired after `take_snapshot()` completes, allowing hook providers to add custom data to the snapshot's `app_data` field:

```python
class CustomDataHook(HookProvider):
    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(SnapshotCreatedEvent, self.on_snapshot_created)

    def on_snapshot_created(self, event: SnapshotCreatedEvent) -> None:
        event.snapshot["app_data"]["custom_key"] = "custom_value"
```

### Snapshottable Protocol

A `Snapshottable` protocol is provided for type-safe operations on objects that support snapshots:

```python
@runtime_checkable
class Snapshottable(Protocol):
    def take_snapshot(
        self,
        app_data: dict[str, Any] | None = None,
        include: SnapshotPreset | list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Snapshot: ...

    def load_snapshot(self, snapshot: Snapshot) -> None: ...
```

### FileSystemPersister

A `FileSystemPersister` helper class is provided for basic persistence:

```python
from strands.agent.snapshot import FileSystemPersister

# Save a snapshot
persister = FileSystemPersister(path="checkpoints/snapshot.json")
persister.save(agent.take_snapshot(include="session"))

# Load a snapshot
snapshot = persister.load()
agent.load_snapshot(snapshot)
```

### Future Concerns

- Snapshotting for MultiAgent constructs - Snapshot is designed in a way that the snapshot could be reused for multi-agent with a similar api
- Providing additional storage APIs for snapshot CRUD operations (database, S3, etc.)
- Providing APIs to customize serialization formats
- Async support for SnapshotCreatedEvent hook

## Developer Experience

### Evaluations via Rewind and Replay

```python
agent = Agent(tools=[tool1, tool2])
snapshot = agent.take_snapshot(include="session")

result1 = agent("What is the weather?")

# ...

agent2 = Agent(tools=[tool3, tool4])
agent2.load_snapshot(snapshot)
result2 = agent2("What is the weather?")
# ... 
# Human/manual evaluation if one outcome was better than the other
# ...
```

### Advanced Context Management

```python
agent = Agent(conversation_manager=CompactingConversationManager())
snapshot = agent.take_snapshot(include="session", app_data={"checkpoint": "before_long_task"})

# ... later ...
later_agent = Agent(conversation_manager=CompactingConversationManager())
later_agent.load_snapshot(snapshot)
```

### Persisting Snapshots

```python
from strands.agent.snapshot import FileSystemPersister

agent = Agent(tools=[tool1, tool2])
agent("Remember that my favorite color is orange.")

# Save to file
snapshot = agent.take_snapshot(include="session", app_data={"user_id": "123"})
FileSystemPersister("snapshot.json").save(snapshot)

# Later, restore from file
snapshot = FileSystemPersister("snapshot.json").load()

agent = Agent(tools=[tool1, tool2])
agent.load_snapshot(snapshot)
agent("What is my favorite color?")  # "Your favorite color is orange."
```

### Selective Field Inclusion

```python
# Include only messages and state
snapshot = agent.take_snapshot(include=["messages", "state"])

# Use session preset but exclude interrupt_state
snapshot = agent.take_snapshot(include="session", exclude=["interrupt_state"])

# Include everything except system_prompt (equivalent to include="session")
snapshot = agent.take_snapshot(exclude=["system_prompt"])
```

### Edge cases

Restoring runtime behavior (e.g., tools) is explicitly not supported:

```python
agent1 = Agent(tools=[tool1, tool2])
snapshot = agent1.take_snapshot(include="session")
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

Yes — Implemented in commits 571c476f, 301c3cd0, and 24cac465.
