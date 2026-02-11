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
            "data": {},
            "app_data": {},
        }

        assert snapshot["type"] == "agent"
        assert snapshot["version"] == "1.0"
        assert snapshot["timestamp"] == "2024-01-01T00:00:00+00:00"
        assert snapshot["data"] == {}
        assert snapshot["app_data"] == {}

    def test_snapshot_is_json_serializable(self):
        """Verify Snapshot can be serialized to JSON."""
        snapshot: Snapshot = {
            "type": "agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "data": {"messages": [], "state": {"key": "value"}},
            "app_data": {"user_id": "123", "session_name": "test"},
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

        snapshot = agent.take_snapshot(include="session")

        assert snapshot["type"] == "agent"
        assert snapshot["version"] == "1.0"
        assert "timestamp" in snapshot
        assert snapshot["data"]["messages"] == []
        assert snapshot["data"]["state"] == {}
        assert snapshot["app_data"] == {}

    def test_take_snapshot_with_messages(self):
        """Take snapshot of agent with conversation history."""
        agent = Agent(
            model=MockedModelProvider([{"role": "assistant", "content": [{"text": "Hello!"}]}]),
        )
        agent("Hi")

        snapshot = agent.take_snapshot(include="session")

        assert len(snapshot["data"]["messages"]) == 2
        assert snapshot["data"]["messages"][0]["content"][0]["text"] == "Hi"
        assert snapshot["data"]["messages"][1]["content"][0]["text"] == "Hello!"

    def test_take_snapshot_with_agent_state(self):
        """Take snapshot preserves agent state."""
        agent = Agent(model=MockedModelProvider([]), state={"counter": 42, "name": "test"})

        snapshot = agent.take_snapshot(include="session")

        assert snapshot["data"]["state"] == {"counter": 42, "name": "test"}

    def test_take_snapshot_with_app_data(self):
        """Take snapshot with application-provided app_data."""
        agent = Agent(model=MockedModelProvider([]))

        snapshot = agent.take_snapshot(include="session", app_data={"user_id": "123", "session": "test"})

        assert snapshot["app_data"] == {"user_id": "123", "session": "test"}

    def test_take_snapshot_captures_conversation_manager_state(self):
        """Take snapshot captures conversation manager state."""
        agent = Agent(model=MockedModelProvider([]))

        snapshot = agent.take_snapshot(include="session")

        assert "conversation_manager_state" in snapshot["data"]
        assert "__name__" in snapshot["data"]["conversation_manager_state"]

    def test_take_snapshot_captures_interrupt_state(self):
        """Take snapshot captures interrupt state."""
        agent = Agent(model=MockedModelProvider([]))

        snapshot = agent.take_snapshot(include="session")

        assert "interrupt_state" in snapshot["data"]
        assert snapshot["data"]["interrupt_state"]["activated"] is False

    def test_take_snapshot_timestamp_is_iso_format(self):
        """Verify timestamp is ISO 8601 format."""
        from datetime import datetime

        agent = Agent(model=MockedModelProvider([]))
        snapshot = agent.take_snapshot(include="session")

        # Should not raise ValueError if valid ISO format
        datetime.fromisoformat(snapshot["timestamp"])


class TestIncludeExcludeParameters:
    """Tests for include/exclude parameter functionality."""

    def test_include_session_preset(self):
        """Session preset includes messages, state, conversation_manager_state, interrupt_state."""
        agent = Agent(model=MockedModelProvider([]), state={"key": "value"}, system_prompt="Test prompt")

        snapshot = agent.take_snapshot(include="session")

        assert "messages" in snapshot["data"]
        assert "state" in snapshot["data"]
        assert "conversation_manager_state" in snapshot["data"]
        assert "interrupt_state" in snapshot["data"]
        assert "system_prompt" not in snapshot["data"]

    def test_include_specific_fields(self):
        """Include only specific fields."""
        agent = Agent(model=MockedModelProvider([]), state={"key": "value"})

        snapshot = agent.take_snapshot(include=["messages", "state"])

        assert "messages" in snapshot["data"]
        assert "state" in snapshot["data"]
        assert "conversation_manager_state" not in snapshot["data"]
        assert "interrupt_state" not in snapshot["data"]

    def test_exclude_fields_from_session(self):
        """Exclude specific fields from session preset."""
        agent = Agent(model=MockedModelProvider([]))

        snapshot = agent.take_snapshot(include="session", exclude=["interrupt_state"])

        assert "messages" in snapshot["data"]
        assert "state" in snapshot["data"]
        assert "conversation_manager_state" in snapshot["data"]
        assert "interrupt_state" not in snapshot["data"]

    def test_exclude_only(self):
        """Exclude fields from all available fields."""
        agent = Agent(model=MockedModelProvider([]), system_prompt="Test")

        snapshot = agent.take_snapshot(exclude=["system_prompt"])

        assert "messages" in snapshot["data"]
        assert "state" in snapshot["data"]
        assert "system_prompt" not in snapshot["data"]

    def test_include_system_prompt(self):
        """Include system_prompt in snapshot."""
        agent = Agent(model=MockedModelProvider([]), system_prompt="You are a helpful assistant.")

        snapshot = agent.take_snapshot(include=["system_prompt"])

        assert snapshot["data"]["system_prompt"] == "You are a helpful assistant."

    def test_require_include_or_exclude(self):
        """Raise error if neither include nor exclude is specified."""
        agent = Agent(model=MockedModelProvider([]))

        with pytest.raises(ValueError, match="Either 'include' or 'exclude' must be specified"):
            agent.take_snapshot()

    def test_invalid_include_field_raises(self):
        """Invalid field names in include raise error."""
        agent = Agent(model=MockedModelProvider([]))

        with pytest.raises(ValueError, match="Invalid snapshot fields"):
            agent.take_snapshot(include=["invalid_field"])

    def test_invalid_exclude_field_raises(self):
        """Invalid field names in exclude raise error."""
        agent = Agent(model=MockedModelProvider([]))

        with pytest.raises(ValueError, match="Invalid exclude fields"):
            agent.take_snapshot(include="session", exclude=["invalid_field"])


class TestSystemPromptSnapshot:
    """Tests for system_prompt snapshotting."""

    def test_system_prompt_captured_when_included(self):
        """System prompt is captured when explicitly included."""
        agent = Agent(model=MockedModelProvider([]), system_prompt="Be helpful")

        snapshot = agent.take_snapshot(include=["system_prompt", "messages"])

        assert snapshot["data"]["system_prompt"] == "Be helpful"

    def test_system_prompt_restored_when_in_snapshot(self):
        """System prompt is restored when present in snapshot."""
        agent = Agent(model=MockedModelProvider([]), system_prompt="Original prompt")
        snapshot: Snapshot = {
            "type": "agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "data": {
                "system_prompt": "Restored prompt",
            },
            "app_data": {},
        }

        agent.load_snapshot(snapshot)

        assert agent.system_prompt == "Restored prompt"

    def test_system_prompt_not_modified_when_not_in_snapshot(self):
        """System prompt is unchanged when not in snapshot."""
        agent = Agent(model=MockedModelProvider([]), system_prompt="Original prompt")
        snapshot: Snapshot = {
            "type": "agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "data": {
                "messages": [],
            },
            "app_data": {},
        }

        agent.load_snapshot(snapshot)

        assert agent.system_prompt == "Original prompt"


class TestAgentLoadSnapshot:
    """Tests for Agent.load_snapshot() method."""

    def test_load_snapshot_restores_messages(self):
        """Load snapshot restores message history."""
        agent = Agent(model=MockedModelProvider([]))
        snapshot: Snapshot = {
            "type": "agent",
            "version": "1.0",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "data": {
                "messages": [{"role": "user", "content": [{"text": "Hello"}]}],
                "state": {},
                "conversation_manager_state": {
                    "__name__": "SlidingWindowConversationManager",
                    "removed_message_count": 0,
                },
                "interrupt_state": {"interrupts": {}, "context": {}, "activated": False},
            },
            "app_data": {},
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
            "data": {
                "messages": [],
                "state": {"counter": 42, "name": "restored"},
                "conversation_manager_state": {
                    "__name__": "SlidingWindowConversationManager",
                    "removed_message_count": 0,
                },
                "interrupt_state": {"interrupts": {}, "context": {}, "activated": False},
            },
            "app_data": {},
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
            "data": {},
            "app_data": {},
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
        snapshot = agent.take_snapshot(include="session", app_data={"checkpoint": "before_modification"})

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
        snapshot = agent.take_snapshot(include="session")

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
        snapshot = agent.take_snapshot(include="session", app_data={"test": "persistence"})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            persister = FileSystemPersister(path=path)
            persister.save(snapshot)

            loaded = persister.load()

            assert loaded["type"] == snapshot["type"]
            assert loaded["version"] == snapshot["version"]
            assert loaded["data"] == snapshot["data"]
            assert loaded["app_data"] == snapshot["app_data"]
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
        snapshot = agent.take_snapshot(include="session")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "snapshot.json"
            persister = FileSystemPersister(path=str(path))

            persister.save(snapshot)

            assert path.exists()
            loaded = persister.load()
            assert loaded["type"] == snapshot["type"]


class TestAppDataPreservation:
    """Tests for application app_data preservation."""

    def test_app_data_not_modified_by_strands(self):
        """Verify Strands does not modify application app_data."""
        original_app_data = {
            "user_id": "user_123",
            "session_name": "test_session",
            "custom_data": {"nested": "value", "list": [1, 2, 3]},
        }

        agent = Agent(model=MockedModelProvider([]))
        snapshot = agent.take_snapshot(include="session", app_data=original_app_data)

        # Load snapshot and take another
        agent.load_snapshot(snapshot)
        # app_data is not carried over on load - it's application-owned

        # Verify original app_data in snapshot is unchanged
        assert snapshot["app_data"] == original_app_data

    def test_app_data_survives_persistence(self):
        """Verify app_data survives persistence round-trip."""
        app_data = {"complex": {"nested": {"data": [1, 2, 3]}}}

        agent = Agent(model=MockedModelProvider([]))
        snapshot = agent.take_snapshot(include="session", app_data=app_data)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            persister = FileSystemPersister(path=path)
            persister.save(snapshot)
            loaded = persister.load()

            assert loaded["app_data"] == app_data
        finally:
            Path(path).unlink(missing_ok=True)


class TestSnapshotCreatedHook:
    """Tests for SnapshotCreatedEvent hook."""

    def test_snapshot_created_hook_is_fired(self):
        """Verify SnapshotCreatedEvent hook is fired after taking snapshot."""
        from strands.hooks import HookProvider, HookRegistry, SnapshotCreatedEvent

        hook_called = []

        class TestHook(HookProvider):
            def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
                registry.add_callback(SnapshotCreatedEvent, self.on_snapshot_created)

            def on_snapshot_created(self, event: SnapshotCreatedEvent) -> None:
                hook_called.append(True)
                event.snapshot["app_data"]["hook_added"] = "test_value"

        agent = Agent(model=MockedModelProvider([]), hooks=[TestHook()])
        snapshot = agent.take_snapshot(include="session")

        assert len(hook_called) == 1
        assert snapshot["app_data"]["hook_added"] == "test_value"

    def test_hook_can_add_custom_data_to_app_data(self):
        """Verify hooks can add custom data to snapshot app_data."""
        from strands.hooks import HookProvider, HookRegistry, SnapshotCreatedEvent

        class CustomDataHook(HookProvider):
            def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
                registry.add_callback(SnapshotCreatedEvent, self.on_snapshot_created)

            def on_snapshot_created(self, event: SnapshotCreatedEvent) -> None:
                event.snapshot["app_data"]["custom_key"] = "custom_value"
                event.snapshot["app_data"]["timestamp_added"] = "2024-01-01"

        agent = Agent(model=MockedModelProvider([]), hooks=[CustomDataHook()])
        snapshot = agent.take_snapshot(include="session", app_data={"original": "data"})

        assert snapshot["app_data"]["original"] == "data"
        assert snapshot["app_data"]["custom_key"] == "custom_value"
        assert snapshot["app_data"]["timestamp_added"] == "2024-01-01"
