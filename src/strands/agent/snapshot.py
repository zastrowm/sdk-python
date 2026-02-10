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

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TypedDict, runtime_checkable

logger = logging.getLogger(__name__)

# Current snapshot version for forward compatibility
SNAPSHOT_VERSION = "1.0"


class Snapshot(TypedDict):
    """Point-in-time capture of agent state.

    A Snapshot captures the complete evolving state of an agent at a specific moment.
    It is designed for easy JSON serialization and supports both Strands-managed state
    and application-owned metadata.

    Attributes:
        type: String identifying the snapshot type (e.g., "agent"). Strands-owned.
        version: Version string for forward compatibility. Strands-owned.
        timestamp: ISO 8601 timestamp of when the snapshot was taken. Strands-owned.
        state: Dict containing the agent's evolving state. Strands-owned, opaque to callers.
        metadata: Dict for application-owned data. Strands does not read or modify this.

    Example:
        ```python
        snapshot = agent.take_snapshot(metadata={"user_id": "123"})
        # Later...
        agent.load_snapshot(snapshot)
        ```
    """

    type: str
    version: str
    timestamp: str
    state: dict[str, Any]
    metadata: dict[str, Any]


@runtime_checkable
class Snapshottable(Protocol):
    """Protocol for objects that support snapshot operations.

    This protocol defines the contract for taking and loading snapshots,
    enabling future extensibility to other types (multi-agent, etc.).

    Example:
        ```python
        def checkpoint(obj: Snapshottable, path: str) -> None:
            snapshot = obj.take_snapshot()
            FileSystemPersister(path).save(snapshot)
        ```
    """

    def take_snapshot(self, metadata: dict[str, Any] | None = None) -> Snapshot:
        """Capture current state as a snapshot.

        Args:
            metadata: Optional application-owned data to include in the snapshot.
                Strands does not read or modify this data.

        Returns:
            A Snapshot containing the complete current state.
        """
        ...

    def load_snapshot(self, snapshot: Snapshot) -> None:
        """Restore state from a snapshot.

        Args:
            snapshot: The snapshot to restore from.

        Raises:
            ValueError: If the snapshot type doesn't match or is invalid.
        """
        ...


class FileSystemPersister:
    """Persistence helper for saving and loading snapshots from the filesystem.

    This class provides a simple pattern for persisting snapshots to JSON files.
    It handles JSON serialization/deserialization and creates parent directories
    as needed.

    Example:
        ```python
        # Save a snapshot
        persister = FileSystemPersister(path="checkpoints/snapshot.json")
        persister.save(agent.take_snapshot())

        # Load a snapshot
        snapshot = persister.load()
        agent.load_snapshot(snapshot)
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
            snapshot = agent.take_snapshot()
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
            agent.load_snapshot(snapshot)
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
