"""Tests for Agent snapshot functionality.

These tests validate the low-level snapshot API for capturing and restoring
agent state at specific points in time.
"""

import json
import tempfile
from pathlib import Path

import pytest

from strands import Agent, tool
from strands.agent.snapshot import FileSystemPersister, Snapshot, Snapshottable

from ...fixtures.mocked_model_provider import MockedModelProvider


class TestSnapshotTypedDict:
    """Tests for Snapshot TypedDict structure."""

    def test_snapshot_has_required_fields(self):
        """Verify Snapshot TypedDict has all required fields."""
        snapshot: Snapshot = {
            "type": "agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "state": {},
            "metadata": {},
        }

        assert snapshot["type"] == "agent"
        assert snapshot["version"] == "1.0"
        assert snapshot["timestamp"] == "2024-01-01T00:00:00+00:00"
        assert snapshot["state"] == {}
        assert snapshot["metadata"] == {}

    def test_snapshot_is_json_serializable(self):
        """Verify Snapshot can be serialized to JSON."""
        snapshot: Snapshot = {
            "type": "agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "state": {"messages": [], "agent_state": {"key": "value"}},
            "metadata": {"user_id": "123", "session_name": "test"},
        }

        json_str = json.dumps(snapshot)
        restored = json.loads(json_str)

        assert restored == snapshot


class TestSnapshottableProtocol:
    """Tests for Snapshottable protocol compliance."""

    def test_agent_is_snapshottable(self):
        """Verify Agent implements Snapshottable protocol."""
        agent = Agent(model=MockedModelProvider([]))

        assert isinstance(agent, Snapshottable)
        assert hasattr(agent, "take_snapshot")
        assert hasattr(agent, "load_snapshot")
        assert callable(agent.take_snapshot)
        assert callable(agent.load_snapshot)


class TestAgentTakeSnapshot:
    """Tests for Agent.take_snapshot() method."""

    def test_take_snapshot_empty_agent(self):
        """Take snapshot of agent with no messages."""
        agent = Agent(model=MockedModelProvider([]))

        snapshot = agent.take_snapshot()

        assert snapshot["type"] == "agent"
        assert snapshot["version"] == "1.0"
        assert "timestamp" in snapshot
        assert snapshot["state"]["messages"] == []
        assert snapshot["state"]["agent_state"] == {}
        assert snapshot["metadata"] == {}

    def test_take_snapshot_with_messages(self):
        """Take snapshot of agent with conversation history."""
        agent = Agent(
            model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Hello!"}]}]),
        )
        agent("Hi")

        snapshot = agent.take_snapshot()

        assert len(snapshot["state"]["messages"]) == 2
        assert snapshot["state"]["messages"][0]["content"][0]["text"] == "Hi"
        assert snapshot["state"]["messages"][1]["content"][0]["text"] == "Hello!"

    def test_take_snapshot_with_agent_state(self):
        """Take snapshot preserves agent state."""
        agent = Agent(model=MockedModelProvider([]), state={"counter": 42, "name": "test"})

        snapshot = agent.take_snapshot()

        assert snapshot["state"]["agent_state"] == {"counter": 42, "name": "test"}

    def test_take_snapshot_with_metadata(self):
        """Take snapshot with application-provided metadata."""
        agent = Agent(model=MockedModelProvider([]))

        snapshot = agent.take_snapshot(metadata={"user_id": "123", "session": "test"})

        assert snapshot["metadata"] == {"user_id": "123", "session": "test"}

    def test_take_snapshot_captures_conversation_manager_state(self):
        """Take snapshot captures conversation manager state."""
        agent = Agent(model=MockedModelProvider([]))

        snapshot = agent.take_snapshot()

        assert "conversation_manager_state" in snapshot["state"]
        assert "__name__" in snapshot["state"]["conversation_manager_state"]

    def test_take_snapshot_captures_interrupt_state(self):
        """Take snapshot captures interrupt state."""
        agent = Agent(model=MockedModelProvider([]))

        snapshot = agent.take_snapshot()

        assert "interrupt_state" in snapshot["state"]
        assert snapshot["state"]["interrupt_state"]["activated"] is False

    def test_take_snapshot_timestamp_is_iso_format(self):
        """Verify timestamp is ISO 8601 format."""
        from datetime import datetime

        agent = Agent(model=MockedModelProvider([]))
        snapshot = agent.take_snapshot()

        # Should not raise ValueError if valid ISO format
        datetime.fromisoformat(snapshot["timestamp"])


class TestAgentLoadSnapshot:
    """Tests for Agent.load_snapshot() method."""

    def test_load_snapshot_restores_messages(self):
        """Load snapshot restores message history."""
        agent = Agent(model=MockedModelProvider([]))
        snapshot: Snapshot = {
            "type": "agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "state": {
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
                "agent_state": {},
                "conversation_manager_state": {
                    "__name__": "SlidingWindowConversationManager",
                    "removed_message_count": 0,
                },
                "interrupt_state": {"interrupts": {}, "context": {}, "activated": False},
            },
            "metadata": {},
        }

        agent.load_snapshot(snapshot)

        assert len(agent.messages) == 1
        assert agent.messages[0]["content"][0]["text"] == "Hello"

    def test_load_snapshot_restores_agent_state(self):
        """Load snapshot restores agent state."""
        agent = Agent(model=MockedModelProvider([]))
        snapshot: Snapshot = {
            "type": "agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "state": {
                "messages": [],
                "agent_state": {"counter": 42, "name": "restored"},
                "conversation_manager_state": {
                    "__name__": "SlidingWindowConversationManager",
                    "removed_message_count": 0,
                },
                "interrupt_state": {"interrupts": {}, "context": {}, "activated": False},
            },
            "metadata": {},
        }

        agent.load_snapshot(snapshot)

        assert agent.state.get("counter") == 42
        assert agent.state.get("name") == "restored"

    def test_load_snapshot_raises_on_type_mismatch(self):
        """Load snapshot raises error if snapshot type doesn't match."""
        agent = Agent(model=MockedModelProvider([]))
        snapshot: Snapshot = {
            "type": "multi-agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "state": {},
            "metadata": {},
        }

        with pytest.raises(ValueError, match="snapshot type"):
            agent.load_snapshot(snapshot)


class TestSnapshotRoundTrip:
    """Tests for complete snapshot round-trip scenarios."""

    def test_round_trip_preserves_state(self):
        """Take snapshot, modify agent, load snapshot restores original state."""
        agent = Agent(
            model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Response 1"}]}]),
            state={"counter": 1},
        )
        agent("First message")

        # Take snapshot at this point
        snapshot = agent.take_snapshot(metadata={"checkpoint": "before_modification"})

        # Modify agent state
        agent.state.set("counter", 999)
        agent.messages.append({"role": "user", "content": [{"text": "Second message"}]})

        # Verify state was modified
        assert agent.state.get("counter") == 999
        assert len(agent.messages) == 3

        # Load snapshot to restore original state
        agent.load_snapshot(snapshot)

        # Verify original state is restored
        assert agent.state.get("counter") == 1
        assert len(agent.messages) == 2

    def test_round_trip_with_tool_calls(self):
        """Snapshot preserves state across tool invocations."""

        @tool
        def increment_counter(agent: Agent):
            """Increment the counter in agent state."""
            current = agent.state.get("counter") or 0
            agent.state.set("counter", current + 1)

        agent = Agent(
            model=MockedModelProvider(
                [
                    {
                        "role": "assistant",
                        "content": [{"toolUse": {"name": "increment_counter", "toolUseId": "1", "input": {}}}],
                    },
                    {"role": "assistant", "content": [{"text": "Counter incremented"}]},
                ]
            ),
            tools=[increment_counter],
            state={"counter": 0},
        )

        # Invoke agent to trigger tool
        agent("Increment counter")

        # Verify tool was called
        assert agent.state.get("counter") == 1

        # Take snapshot
        snapshot = agent.take_snapshot()

        # Load into fresh agent
        agent2 = Agent(model=MockedModelProvider([]), tools=[increment_counter], state={"counter": 0})
        agent2.load_snapshot(snapshot)

        assert agent2.state.get("counter") == 1
        assert len(agent2.messages) == len(agent.messages)


class TestFileSystemPersister:
    """Tests for FileSystemPersister persistence helper."""

    def test_save_and_load_snapshot(self):
        """Save snapshot to file and load it back."""
        agent = Agent(
            model=MockedModelProvider([]),
            messages=[{"role": "user", "content": [{"text": "Hello"}]}],
            state={"key": "value"},
        )
        snapshot = agent.take_snapshot(metadata={"test": "persistence"})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            persister = FileSystemPersister(path=path)
            persister.save(snapshot)

            loaded = persister.load()

            assert loaded["type"] == snapshot["type"]
            assert loaded["version"] == snapshot["version"]
            assert loaded["state"] == snapshot["state"]
            assert loaded["metadata"] == snapshot["metadata"]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_nonexistent_file_raises(self):
        """Loading from nonexistent file raises error."""
        persister = FileSystemPersister(path="/nonexistent/path/snapshot.json")

        with pytest.raises(FileNotFoundError):
            persister.load()

    def test_persister_creates_parent_directories(self):
        """Persister creates parent directories if they don't exist."""
        agent = Agent(model=MockedModelProvider([]))
        snapshot = agent.take_snapshot()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "snapshot.json"
            persister = FileSystemPersister(path=str(path))

            persister.save(snapshot)

            assert path.exists()
            loaded = persister.load()
            assert loaded["type"] == snapshot["type"]


class TestMetadataPreservation:
    """Tests for application metadata preservation."""

    def test_metadata_not_modified_by_strands(self):
        """Verify Strands does not modify application metadata."""
        original_metadata = {
            "user_id": "user_123",
            "session_name": "test_session",
            "custom_data": {"nested": "value", "list": [1, 2, 3]},
        }

        agent = Agent(model=MockedModelProvider([]))
        snapshot = agent.take_snapshot(metadata=original_metadata)

        # Load snapshot and take another
        agent.load_snapshot(snapshot)
        # Metadata is not carried over on load - it's application-owned

        # Verify original metadata in snapshot is unchanged
        assert snapshot["metadata"] == original_metadata

    def test_metadata_survives_persistence(self):
        """Verify metadata survives persistence round-trip."""
        metadata = {"complex": {"nested": {"data": [1, 2, 3]}}}

        agent = Agent(model=MockedModelProvider([]))
        snapshot = agent.take_snapshot(metadata=metadata)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            persister = FileSystemPersister(path=path)
            persister.save(snapshot)
            loaded = persister.load()

            assert loaded["metadata"] == metadata
        finally:
            Path(path).unlink(missing_ok=True)
