"""Low-level snapshot API for capturing and restoring agent state.

This module provides the snapshot infrastructure for point-in-time capture and restore
of agent state. Snapshots enable:
- Point-in-time capture: Save complete agent state at any moment
- Arbitrary restore: Restore to any previously captured state
- Context management: Enable advanced strategies with state modifications

The snapshot API is independent of SessionManager and provides a different capability:
SessionManager works incrementally, recording messages as they happen, while snapshots
capture the complete state at a specific point in time.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from .agent import Agent

logger = logging.getLogger(__name__)

# Current snapshot version for forward compatibility
SNAPSHOT_VERSION = "1.0"

# Named preset for include parameter
SnapshotPreset = Literal["session"]

# All available snapshot fields
ALL_SNAPSHOT_FIELDS = frozenset(["messages", "state", "conversation_manager_state", "interrupt_state", "system_prompt"])

# Fields included in the "session" preset (excludes system_prompt)
SESSION_PRESET_FIELDS = frozenset(["messages", "state", "conversation_manager_state", "interrupt_state"])


class Snapshot(TypedDict):
    """Point-in-time capture of agent state.

    A Snapshot captures the complete evolving state of an agent at a specific moment.
    It is designed for easy JSON serialization and supports both Strands-managed data
    and application-owned data.

    Attributes:
        type: String identifying the snapshot type (e.g., "agent"). Strands-owned.
        version: Version string for forward compatibility. Strands-owned.
        timestamp: ISO 8601 timestamp of when the snapshot was taken. Strands-owned.
        data: Dict containing the agent's evolving state. This is Strands-managed and
            opaque to callers - applications should not rely on its internal structure.
        app_data: Dict for application-owned data. Strands does not read or modify this.
            Use this field to store application-specific context like user IDs, session
            names, checkpoints, or any custom data your application needs.

    Example:
        ```python
        snapshot = agent.snapshots.take(app_data={"user_id": "123"})
        # Later...
        agent.snapshots.load(snapshot)
        ```
    """

    type: str
    version: str
    timestamp: str
    data: dict[str, Any]
    app_data: dict[str, Any]


class Snapshotter:
    """Manages snapshot operations for an agent.

    The Snapshotter provides methods to capture and restore agent state at specific
    points in time. It can be configured with default include/exclude settings that
    apply to all snapshot operations, while still allowing per-call overrides.

    Example:
        ```python
        # Create agent with default snapshot settings
        snapshotter = Snapshotter(include="session")
        agent = Agent(snapshots=snapshotter)

        # Take snapshot using defaults
        snapshot = agent.snapshots.take()

        # Override defaults for specific snapshot
        snapshot = agent.snapshots.take(include=["messages", "state"])

        # Restore from snapshot
        agent.snapshots.load(snapshot)
        ```

    Attributes:
        include: Default fields to include in snapshots.
        exclude: Default fields to exclude from snapshots.
    """

    def __init__(
        self,
        include: SnapshotPreset | list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        """Initialize the Snapshotter with default settings.

        Args:
            include: Default fields to include in snapshots. Can be:
                - "session": Preset that includes messages, state, conversation_manager_state,
                  and interrupt_state (excludes system_prompt)
                - A list of field names to include
            exclude: Default fields to exclude from snapshots. Applied after include.

        Example:
            ```python
            # Use session preset by default
            snapshotter = Snapshotter(include="session")

            # Exclude interrupt_state by default
            snapshotter = Snapshotter(include="session", exclude=["interrupt_state"])
            ```
        """
        self._default_include = include
        self._default_exclude = exclude
        self._agent: Agent | None = None

    def _bind(self, agent: Agent) -> None:
        """Bind this snapshotter to an agent.

        Args:
            agent: The agent to bind to.
        """
        self._agent = agent

    def take(
        self,
        app_data: dict[str, Any] | None = None,
        include: SnapshotPreset | list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Snapshot:
        """Capture current agent state as a snapshot.

        Creates a point-in-time capture of the agent's evolving state. The fields
        included in the snapshot are controlled by the `include` and `exclude` parameters,
        falling back to the defaults configured on the Snapshotter.

        Available fields:
        - messages: Conversation history
        - state: Custom application state (agent.state)
        - conversation_manager_state: Internal state of the conversation manager
        - interrupt_state: State of any active interrupts
        - system_prompt: The agent's system prompt

        Args:
            app_data: Optional application-owned data to include in the snapshot.
                Strands does not read or modify this data. Use it to store
                application-specific context like user IDs, session names, etc.
            include: Fields to include in the snapshot. Overrides the default.
                Can be "session" preset or a list of field names.
            exclude: Fields to exclude from the snapshot. Overrides the default.

        Returns:
            A Snapshot containing the captured state.

        Raises:
            RuntimeError: If the snapshotter is not bound to an agent.
            ValueError: If neither include nor exclude is specified (and no defaults set).

        Example:
            ```python
            # Take a snapshot with defaults
            snapshot = agent.snapshots.take()

            # Take a snapshot with app_data
            snapshot = agent.snapshots.take(app_data={"checkpoint": "before_update"})

            # Override defaults for this snapshot
            snapshot = agent.snapshots.take(include=["messages", "state"])
            ```
        """
        if self._agent is None:
            raise RuntimeError("Snapshotter is not bound to an agent")

        from ..hooks import SnapshotCreatedEvent

        # Use provided values or fall back to defaults
        effective_include = include if include is not None else self._default_include
        effective_exclude = exclude if exclude is not None else self._default_exclude

        logger.debug("agent_id=<%s> | taking snapshot", self._agent.agent_id)

        # Resolve which fields to include
        fields = resolve_snapshot_fields(effective_include, effective_exclude)

        # Build the data dict with only the requested fields
        data: dict[str, Any] = {}

        if "messages" in fields:
            data["messages"] = list(self._agent.messages)

        if "state" in fields:
            data["state"] = self._agent.state.get()

        if "conversation_manager_state" in fields:
            data["conversation_manager_state"] = self._agent.conversation_manager.get_state()

        if "interrupt_state" in fields:
            data["interrupt_state"] = self._agent._interrupt_state.to_dict()

        if "system_prompt" in fields:
            data["system_prompt"] = self._agent._system_prompt

        snapshot: Snapshot = {
            "type": "agent",
            "version": SNAPSHOT_VERSION,
            "timestamp": create_timestamp(),
            "data": data,
            "app_data": app_data or {},
        }

        # TODO: This hook invocation is synchronous. Consider adding async support
        # for hooks that need to perform async operations when a snapshot is created.
        # Fire hook to allow custom data to be added to the snapshot
        self._agent.hooks.invoke_callbacks(SnapshotCreatedEvent(agent=self._agent, snapshot=snapshot))

        logger.debug(
            "agent_id=<%s>, message_count=<%d> | snapshot taken", self._agent.agent_id, len(self._agent.messages)
        )
        return snapshot

    async def take_async(
        self,
        app_data: dict[str, Any] | None = None,
        include: SnapshotPreset | list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Snapshot:
        """Capture current agent state as a snapshot asynchronously.

        This is the async version of take(). Currently, the implementation is
        synchronous, but this method is provided for API consistency and future
        async hook support.

        Args:
            app_data: Optional application-owned data to include in the snapshot.
            include: Fields to include in the snapshot. Overrides the default.
            exclude: Fields to exclude from the snapshot. Overrides the default.

        Returns:
            A Snapshot containing the captured state.

        Raises:
            RuntimeError: If the snapshotter is not bound to an agent.
            ValueError: If neither include nor exclude is specified (and no defaults set).
        """
        # Currently synchronous, but provides async API for future hook support
        return self.take(app_data=app_data, include=include, exclude=exclude)

    def load(self, snapshot: Snapshot) -> None:
        """Restore agent state from a snapshot.

        Restores the agent's evolving state from a previously captured snapshot.
        Only fields that were included when the snapshot was taken will be restored.

        Restorable fields:
        - messages: Replaces current conversation history
        - state: Replaces current application state
        - conversation_manager_state: Restores conversation manager state
        - interrupt_state: Restores interrupt state
        - system_prompt: Restores the system prompt

        Note: Agent definition aspects not included in the snapshot (tools, model,
        conversation_manager instance) are not affected.

        Args:
            snapshot: The snapshot to restore from.

        Raises:
            RuntimeError: If the snapshotter is not bound to an agent.
            ValueError: If the snapshot type is not "agent".

        Example:
            ```python
            # Load a previously saved snapshot
            snapshot = FileSystemPersister("checkpoint.json").load()
            agent.snapshots.load(snapshot)
            ```
        """
        if self._agent is None:
            raise RuntimeError("Snapshotter is not bound to an agent")

        from ..interrupt import _InterruptState
        from .state import AgentState

        logger.debug("agent_id=<%s> | loading snapshot", self._agent.agent_id)

        if snapshot["type"] != "agent":
            raise ValueError(f"snapshot type=<{snapshot['type']}> | expected snapshot type 'agent'")

        data = snapshot["data"]

        # Restore messages
        if "messages" in data:
            self._agent.messages = list(data["messages"])

        # Restore agent state
        if "state" in data:
            self._agent.state = AgentState(data["state"])

        # Restore conversation manager state
        if "conversation_manager_state" in data:
            self._agent.conversation_manager.restore_from_session(data["conversation_manager_state"])

        # Restore interrupt state
        if "interrupt_state" in data:
            self._agent._interrupt_state = _InterruptState.from_dict(data["interrupt_state"])

        # Restore system prompt
        if "system_prompt" in data:
            self._agent.system_prompt = data["system_prompt"]

        logger.debug(
            "agent_id=<%s>, message_count=<%d> | snapshot loaded", self._agent.agent_id, len(self._agent.messages)
        )


class FileSystemPersister:
    """Persistence helper for saving and loading snapshots from the filesystem.

    This class provides a simple pattern for persisting snapshots to JSON files.
    It handles JSON serialization/deserialization and creates parent directories
    as needed.

    Example:
        ```python
        # Save a snapshot
        persister = FileSystemPersister(path="checkpoints/snapshot.json")
        persister.save(agent.snapshots.take())

        # Load a snapshot
        snapshot = persister.load()
        agent.snapshots.load(snapshot)
        ```

    Attributes:
        path: Path to the snapshot file.
    """

    def __init__(self, path: str) -> None:
        """Initialize the persister with a file path.

        Args:
            path: Path where the snapshot will be saved/loaded.
        """
        self.path = Path(path)

    def save(self, snapshot: Snapshot) -> None:
        """Save a snapshot to the filesystem.

        Creates parent directories if they don't exist.

        Args:
            snapshot: The snapshot to save.

        Example:
            ```python
            snapshot = agent.snapshots.take()
            FileSystemPersister("state.json").save(snapshot)
            ```
        """
        logger.debug("path=<%s> | saving snapshot to filesystem", self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        logger.debug("path=<%s> | snapshot saved successfully", self.path)

    def load(self) -> Snapshot:
        """Load a snapshot from the filesystem.

        Returns:
            The loaded snapshot.

        Raises:
            FileNotFoundError: If the snapshot file doesn't exist.

        Example:
            ```python
            snapshot = FileSystemPersister("state.json").load()
            agent.snapshots.load(snapshot)
            ```
        """
        logger.debug("path=<%s> | loading snapshot from filesystem", self.path)
        with open(self.path, encoding="utf-8") as f:
            snapshot: Snapshot = json.load(f)
        logger.debug("path=<%s> | snapshot loaded successfully", self.path)
        return snapshot


def create_timestamp() -> str:
    """Create an ISO 8601 timestamp for the current time.

    Returns:
        ISO 8601 formatted timestamp string.
    """
    return datetime.now(timezone.utc).isoformat()


def resolve_snapshot_fields(
    include: SnapshotPreset | list[str] | None = None,
    exclude: list[str] | None = None,
) -> set[str]:
    """Resolve the final set of fields to include in a snapshot.

    Args:
        include: Fields to include. Can be a preset name or list of field names.
        exclude: Fields to exclude from the included set.

    Returns:
        Set of field names to include in the snapshot.

    Raises:
        ValueError: If neither include nor exclude is specified, or if invalid fields are specified.
    """
    if include is None and exclude is None:
        raise ValueError("Either 'include' or 'exclude' must be specified")

    # Determine base fields from include
    if include is None:
        fields = set(ALL_SNAPSHOT_FIELDS)
    elif include == "session":
        fields = set(SESSION_PRESET_FIELDS)
    elif isinstance(include, list):
        invalid_fields = set(include) - ALL_SNAPSHOT_FIELDS
        if invalid_fields:
            raise ValueError(f"Invalid snapshot fields: {invalid_fields}. Valid fields: {ALL_SNAPSHOT_FIELDS}")
        fields = set(include)
    else:
        raise ValueError(f"Invalid include value: {include}. Must be 'session' or a list of field names.")

    # Apply exclusions
    if exclude:
        invalid_fields = set(exclude) - ALL_SNAPSHOT_FIELDS
        if invalid_fields:
            raise ValueError(f"Invalid exclude fields: {invalid_fields}. Valid fields: {ALL_SNAPSHOT_FIELDS}")
        fields -= set(exclude)

    return fields
