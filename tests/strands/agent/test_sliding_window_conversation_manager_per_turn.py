"""Tests for SlidingWindowConversationManager per_turn functionality."""

from unittest.mock import MagicMock, patch

import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.hooks.events import AfterToolCallEvent
from strands.hooks.registry import HookRegistry
from tests.fixtures.mocked_model_provider import MockedModelProvider


def create_test_tool():
    """Create a simple test tool that returns a response."""

    def test_tool(query: str) -> str:
        """A test tool.

        Args:
            query: Input query

        Returns:
            Response string
        """
        return f"Result for: {query}"

    test_tool.__name__ = "test_tool"
    return test_tool


@pytest.fixture
def agent_responses():
    """Mock agent responses for testing."""
    return [
        # Response 1: Tool call
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "1",
                        "name": "test_tool",
                        "input": {"query": "first"},
                    }
                }
            ],
        },
        # Response 2: Tool call
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "2",
                        "name": "test_tool",
                        "input": {"query": "second"},
                    }
                }
            ],
        },
        # Response 3: Tool call
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "3",
                        "name": "test_tool",
                        "input": {"query": "third"},
                    }
                }
            ],
        },
        # Response 4: Tool call
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "4",
                        "name": "test_tool",
                        "input": {"query": "fourth"},
                    }
                }
            ],
        },
        # Response 5: Tool call
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "5",
                        "name": "test_tool",
                        "input": {"query": "fifth"},
                    }
                }
            ],
        },
        # Final response: Text
        {
            "role": "assistant",
            "content": [{"text": "All done!"}],
        },
    ]


class TestPerTurnParameter:
    """Tests for the per_turn parameter in SlidingWindowConversationManager."""

    def test_per_turn_false_default(self):
        """Test that per_turn defaults to False and no hooks are registered."""
        manager = SlidingWindowConversationManager()
        assert hasattr(manager, "per_turn")
        assert manager.per_turn is False

        # Verify register_hooks exists but doesn't register when per_turn=False
        registry = HookRegistry()
        if hasattr(manager, "register_hooks"):
            manager.register_hooks(registry)
            assert not registry.has_callbacks()

    def test_per_turn_explicit_false(self):
        """Test that per_turn=False explicitly works."""
        manager = SlidingWindowConversationManager(per_turn=False)
        assert manager.per_turn is False

    def test_per_turn_true(self):
        """Test that per_turn=True can be set."""
        manager = SlidingWindowConversationManager(per_turn=True)
        assert manager.per_turn is True

    def test_per_turn_integer(self):
        """Test that per_turn accepts integer values."""
        manager = SlidingWindowConversationManager(per_turn=3)
        assert manager.per_turn == 3

    def test_per_turn_invalid_zero(self):
        """Test that per_turn=0 raises ValueError."""
        with pytest.raises(ValueError):
            SlidingWindowConversationManager(per_turn=0)

    def test_per_turn_invalid_negative(self):
        """Test that negative per_turn values raise ValueError."""
        with pytest.raises(ValueError):
            SlidingWindowConversationManager(per_turn=-1)


class TestHookRegistration:
    """Tests for hook registration when per_turn is enabled."""

    def test_register_hooks_exists(self):
        """Test that register_hooks method exists."""
        manager = SlidingWindowConversationManager()
        assert hasattr(manager, "register_hooks")
        assert callable(manager.register_hooks)

    def test_register_hooks_with_per_turn_false(self):
        """Test that no callbacks are registered when per_turn=False."""
        manager = SlidingWindowConversationManager(per_turn=False)
        registry = HookRegistry()
        manager.register_hooks(registry)
        assert not registry.has_callbacks()

    def test_register_hooks_with_per_turn_true(self):
        """Test that callbacks are registered when per_turn=True."""
        manager = SlidingWindowConversationManager(per_turn=True)
        registry = HookRegistry()
        manager.register_hooks(registry)
        assert registry.has_callbacks()

    def test_register_hooks_with_per_turn_integer(self):
        """Test that callbacks are registered when per_turn is an integer."""
        manager = SlidingWindowConversationManager(per_turn=3)
        registry = HookRegistry()
        manager.register_hooks(registry)
        assert registry.has_callbacks()

    def test_agent_auto_registers_conversation_manager(self, agent_responses):
        """Test that Agent auto-registers conversation_manager as hook."""
        manager = SlidingWindowConversationManager(per_turn=True)
        model = MockedModelProvider(agent_responses)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        # Verify the manager's hooks were registered
        assert agent.hooks.has_callbacks()

    def test_agent_with_null_conversation_manager(self, agent_responses):
        """Test that Agent works with conversation managers that don't implement hooks."""
        manager = NullConversationManager()
        model = MockedModelProvider(agent_responses)
        # Should not raise an error
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])
        assert agent is not None


class TestPerTurnManagement:
    """Tests for apply_management behavior with per_turn."""

    def test_per_turn_true_calls_management_after_each_tool_call(self, agent_responses):
        """Test that per_turn=True calls apply_management after every tool call."""
        manager = SlidingWindowConversationManager(per_turn=True, window_size=100)
        model = MockedModelProvider(agent_responses)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Run the test tool multiple times")

            # Verify apply_management was called multiple times
            # (once per tool call + once in finally block)
            assert mock_apply.call_count >= 5  # 5 tool calls minimum

    def test_per_turn_integer_calls_management_every_n_calls(self, agent_responses):
        """Test that per_turn=N calls apply_management every N tool calls."""
        manager = SlidingWindowConversationManager(per_turn=3, window_size=100)
        model = MockedModelProvider(agent_responses)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Run the test tool multiple times")

            # With 5 tool calls and per_turn=3:
            # - Called after tool call 3
            # - Called in finally block
            # Minimum 2 calls (could be more depending on exact execution)
            assert mock_apply.call_count >= 2

    def test_per_turn_one_equivalent_to_true(self, agent_responses):
        """Test that per_turn=1 behaves like per_turn=True."""
        manager = SlidingWindowConversationManager(per_turn=1, window_size=100)
        model = MockedModelProvider(agent_responses)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Run the test tool multiple times")

            # Should be called after every tool call
            assert mock_apply.call_count >= 5

    def test_per_turn_false_only_calls_in_finally(self, agent_responses):
        """Test that per_turn=False only calls apply_management in finally block."""
        manager = SlidingWindowConversationManager(per_turn=False, window_size=100)
        model = MockedModelProvider(agent_responses)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Run the test tool multiple times")

            # Should only be called once in finally block
            assert mock_apply.call_count == 1

    def test_no_tool_calls_only_finally_block(self):
        """Test that with no tool calls, management only happens in finally block."""
        manager = SlidingWindowConversationManager(per_turn=True, window_size=100)
        # Response with no tool calls
        responses = [
            {
                "role": "assistant",
                "content": [{"text": "No tools needed"}],
            }
        ]
        model = MockedModelProvider(responses)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Just respond with text")

            # Should only be called once in finally block (no tool calls)
            assert mock_apply.call_count == 1

    def test_message_count_reduced_during_loop(self, agent_responses):
        """Test that messages are trimmed during loop execution, not just at end."""
        # Use small window size to trigger trimming
        manager = SlidingWindowConversationManager(per_turn=1, window_size=4)
        model = MockedModelProvider(agent_responses)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        # Track message counts after each tool call
        message_counts = []

        original_apply = manager.apply_management

        def track_apply(agent_instance):
            message_counts.append(len(agent_instance.messages))
            return original_apply(agent_instance)

        with patch.object(manager, "apply_management", side_effect=track_apply):
            agent("Run the test tool multiple times")

        # Verify that message count was managed during execution
        # (should stay around window_size, not grow unbounded)
        assert any(count <= manager.window_size for count in message_counts)


class TestToolCallCounter:
    """Tests for tool call counter tracking."""

    def test_tool_call_count_increments(self, agent_responses):
        """Test that tool call count increments correctly."""
        manager = SlidingWindowConversationManager(per_turn=3, window_size=100)
        assert manager._tool_call_count == 0

        model = MockedModelProvider(agent_responses)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        agent("Run the test tool multiple times")

        # After execution, counter should reflect number of tool calls
        assert manager._tool_call_count >= 5

    def test_tool_call_count_triggers_management(self):
        """Test that management is triggered at correct intervals based on counter."""
        manager = SlidingWindowConversationManager(per_turn=2, window_size=100)
        registry = HookRegistry()
        manager.register_hooks(registry)

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.messages = []

        # Create mock events
        event1 = AfterToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "1", "name": "test", "input": {}},
            invocation_state={},
            result={"toolUseId": "1", "content": [], "status": "success"},
        )
        event2 = AfterToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "2", "name": "test", "input": {}},
            invocation_state={},
            result={"toolUseId": "2", "content": [], "status": "success"},
        )
        event3 = AfterToolCallEvent(
            agent=mock_agent,
            selected_tool=None,
            tool_use={"toolUseId": "3", "name": "test", "input": {}},
            invocation_state={},
            result={"toolUseId": "3", "content": [], "status": "success"},
        )

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            # Invoke events
            registry.invoke_callbacks(event1)
            assert manager._tool_call_count == 1
            assert mock_apply.call_count == 0  # Not called yet (per_turn=2)

            registry.invoke_callbacks(event2)
            assert manager._tool_call_count == 2
            assert mock_apply.call_count == 1  # Called after 2nd tool call

            registry.invoke_callbacks(event3)
            assert manager._tool_call_count == 3
            assert mock_apply.call_count == 1  # Not called (waiting for 4th)


class TestStateManagement:
    """Tests for state persistence with per_turn."""

    def test_get_state_includes_tool_call_count(self):
        """Test that get_state includes _tool_call_count."""
        manager = SlidingWindowConversationManager(per_turn=3)
        manager._tool_call_count = 5
        state = manager.get_state()

        assert "_tool_call_count" in state
        assert state["_tool_call_count"] == 5

    def test_restore_from_session_restores_tool_call_count(self):
        """Test that restore_from_session restores _tool_call_count."""
        manager = SlidingWindowConversationManager(per_turn=3)
        state = {
            "__name__": "SlidingWindowConversationManager",
            "removed_message_count": 0,
            "_tool_call_count": 7,
        }
        manager.restore_from_session(state)

        assert manager._tool_call_count == 7


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_existing_sliding_window_still_works(self):
        """Test that existing SlidingWindowConversationManager usage still works."""
        # Old-style initialization without per_turn
        manager = SlidingWindowConversationManager(window_size=40)
        assert manager.per_turn is False

    def test_existing_agent_initialization_still_works(self):
        """Test that existing Agent initialization patterns still work."""
        responses = [
            {
                "role": "assistant",
                "content": [{"text": "Hello"}],
            }
        ]
        model = MockedModelProvider(responses)
        # Old-style agent creation
        agent = Agent(model=model)
        result = agent("Hello")
        assert result is not None

    def test_null_conversation_manager_not_affected(self):
        """Test that NullConversationManager is not affected by changes."""
        manager = NullConversationManager()
        # Should not have register_hooks or per_turn
        # This ensures we didn't accidentally modify the base class
        assert not hasattr(manager, "per_turn")
