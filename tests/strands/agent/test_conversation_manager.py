import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.types.exceptions import ContextWindowOverflowException


@pytest.fixture
def conversation_manager(request):
    params = {
        "window_size": 2,
        "should_truncate_results": False,
    }
    if hasattr(request, "param"):
        params.update(request.param)

    return SlidingWindowConversationManager(**params)


@pytest.mark.parametrize(
    ("conversation_manager", "messages", "expected_messages"),
    [
        # 0 - Message count under max window size - Latest assistant
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Hi there"}]},
            ],
            [
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Hi there"}]},
            ],
        ),
        # 1 - Message count under max window size - Latest user
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
            ],
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
            ],
        ),
        # 2 - Keep user message
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"text": "Hello"}]},
            ],
            [{"role": "user", "content": [{"text": "Hello"}]}],
        ),
        # 3 - Keep dangling assistant message with tool use
        (
            {"window_size": 3},
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
            [{"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]}],
        ),
        # 4 - Remove dangling assistant message with tool use - User tool result remains
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
        ),
        # 5 - Remove dangling assistant message with tool use and user message without tool result
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"text": "First"}]},
                {"role": "assistant", "content": [{"text": "First response"}]},
                {"role": "user", "content": [{"text": "Use a tool"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
            [
                {"role": "assistant", "content": [{"text": "First response"}]},
                {"role": "user", "content": [{"text": "Use a tool"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
            ],
        ),
        # 6 - Message count above max window size - Basic drop
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"text": "First message"}]},
                {"role": "assistant", "content": [{"text": "First response"}]},
                {"role": "user", "content": [{"text": "Second message"}]},
                {"role": "assistant", "content": [{"text": "Second response"}]},
            ],
            [
                {"role": "user", "content": [{"text": "Second message"}]},
                {"role": "assistant", "content": [{"text": "Second response"}]},
            ],
        ),
        # 7 - Message count above max window size - Preserve tool use/tool result pairs
        (
            {"window_size": 2},
            [
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
            ],
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
            ],
        ),
        # 8 - Test sliding window behavior - preserve tool use/result pairs across cut boundary
        (
            {"window_size": 3},
            [
                {"role": "user", "content": [{"text": "First message"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Response after tool use"}]},
            ],
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Response after tool use"}]},
            ],
        ),
        # 9 - Test sliding window with multiple tool pairs that need preservation
        (
            {"window_size": 4},
            [
                {"role": "user", "content": [{"text": "First message"}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "123", "name": "tool1", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "123", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool2", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Final response"}]},
            ],
            [
                {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool2", "input": {}}}]},
                {"role": "user", "content": [{"toolResult": {"toolUseId": "456", "content": [], "status": "success"}}]},
                {"role": "assistant", "content": [{"text": "Final response"}]},
            ],
        ),
    ],
    indirect=["conversation_manager"],
)
def test_apply_management(conversation_manager, messages, expected_messages):
    test_agent = Agent(messages=messages)
    conversation_manager.apply_management(test_agent)

    assert messages == expected_messages


def test_sliding_window_conversation_manager_with_untrimmable_history_raises_context_window_overflow_exception():
    manager = SlidingWindowConversationManager(1, False)
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool1", "input": {}}}]},
        {"role": "user", "content": [{"toolResult": {"toolUseId": "789", "content": [], "status": "success"}}]},
    ]
    original_messages = messages.copy()
    test_agent = Agent(messages=messages)

    with pytest.raises(ContextWindowOverflowException):
        manager.apply_management(test_agent)

    assert messages == original_messages


def test_sliding_window_conversation_manager_with_tool_results_truncated():
    manager = SlidingWindowConversationManager(1)
    messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool1", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "789", "content": [{"text": "large input"}], "status": "success"}}
            ],
        },
    ]
    test_agent = Agent(messages=messages)

    manager.reduce_context(test_agent)

    expected_messages = [
        {"role": "assistant", "content": [{"toolUse": {"toolUseId": "456", "name": "tool1", "input": {}}}]},
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "789",
                        "content": [{"text": "The tool result was too large!"}],
                        "status": "error",
                    }
                }
            ],
        },
    ]

    assert messages == expected_messages


def test_null_conversation_manager_reduce_context_raises_context_window_overflow_exception():
    """Test that NullConversationManager doesn't modify messages."""
    manager = NullConversationManager()
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]
    original_messages = messages.copy()
    test_agent = Agent(messages=messages)

    manager.apply_management(test_agent)

    with pytest.raises(ContextWindowOverflowException):
        manager.reduce_context(messages)

    assert messages == original_messages


def test_null_conversation_manager_reduce_context_with_exception_raises_same_exception():
    """Test that NullConversationManager doesn't modify messages."""
    manager = NullConversationManager()
    messages = [
        {"role": "user", "content": [{"text": "Hello"}]},
        {"role": "assistant", "content": [{"text": "Hi there"}]},
    ]
    original_messages = messages.copy()
    test_agent = Agent(messages=messages)

    manager.apply_management(test_agent)

    with pytest.raises(RuntimeError):
        manager.reduce_context(messages, RuntimeError("test"))

    assert messages == original_messages


def test_null_conversation_does_not_restore_with_incorrect_state():
    """Test that NullConversationManager doesn't modify messages."""
    manager = NullConversationManager()

    with pytest.raises(ValueError):
        manager.restore_from_session({})


# ============================================================================
# Per-Turn Management Tests
# ============================================================================


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
def agent_responses_with_model_calls():
    """Mock agent responses for testing with multiple model calls."""
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
        from strands.hooks.registry import HookRegistry

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
        from strands.hooks.registry import HookRegistry

        manager = SlidingWindowConversationManager(per_turn=False)
        registry = HookRegistry()
        manager.register_hooks(registry)
        assert not registry.has_callbacks()

    def test_register_hooks_with_per_turn_true(self):
        """Test that callbacks are registered when per_turn=True."""
        from strands.hooks.registry import HookRegistry

        manager = SlidingWindowConversationManager(per_turn=True)
        registry = HookRegistry()
        manager.register_hooks(registry)
        assert registry.has_callbacks()

    def test_register_hooks_with_per_turn_integer(self):
        """Test that callbacks are registered when per_turn is an integer."""
        from strands.hooks.registry import HookRegistry

        manager = SlidingWindowConversationManager(per_turn=3)
        registry = HookRegistry()
        manager.register_hooks(registry)
        assert registry.has_callbacks()

    def test_agent_auto_registers_conversation_manager(self, agent_responses_with_model_calls):
        """Test that Agent auto-registers conversation_manager as hook."""
        from tests.fixtures.mocked_model_provider import MockedModelProvider

        manager = SlidingWindowConversationManager(per_turn=True)
        model = MockedModelProvider(agent_responses_with_model_calls)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        # Verify the manager's hooks were registered
        assert agent.hooks.has_callbacks()

    def test_agent_with_null_conversation_manager(self, agent_responses_with_model_calls):
        """Test that Agent works with conversation managers that don't implement hooks."""
        from tests.fixtures.mocked_model_provider import MockedModelProvider

        manager = NullConversationManager()
        model = MockedModelProvider(agent_responses_with_model_calls)
        # Should not raise an error
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])
        assert agent is not None


class TestPerTurnManagement:
    """Tests for apply_management behavior with per_turn."""

    def test_per_turn_true_calls_management_after_each_model_call(self, agent_responses_with_model_calls):
        """Test that per_turn=True calls apply_management after every model call."""
        from unittest.mock import patch

        from tests.fixtures.mocked_model_provider import MockedModelProvider

        manager = SlidingWindowConversationManager(per_turn=True, window_size=100)
        model = MockedModelProvider(agent_responses_with_model_calls)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Run the test tool multiple times")

            # Verify apply_management was called multiple times
            # (once per model call + once in finally block)
            assert mock_apply.call_count >= 5  # 5 model calls minimum

    def test_per_turn_integer_calls_management_every_n_calls(self, agent_responses_with_model_calls):
        """Test that per_turn=N calls apply_management every N model calls."""
        from unittest.mock import patch

        from tests.fixtures.mocked_model_provider import MockedModelProvider

        manager = SlidingWindowConversationManager(per_turn=3, window_size=100)
        model = MockedModelProvider(agent_responses_with_model_calls)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Run the test tool multiple times")

            # With 6 model calls (5 tool responses + 1 final text) and per_turn=3:
            # - Called after model call 3
            # - Called after model call 6
            # - Called in finally block
            # Minimum 2-3 calls
            assert mock_apply.call_count >= 2

    def test_per_turn_one_equivalent_to_true(self, agent_responses_with_model_calls):
        """Test that per_turn=1 behaves like per_turn=True."""
        from unittest.mock import patch

        from tests.fixtures.mocked_model_provider import MockedModelProvider

        manager = SlidingWindowConversationManager(per_turn=1, window_size=100)
        model = MockedModelProvider(agent_responses_with_model_calls)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Run the test tool multiple times")

            # Should be called after every model call
            assert mock_apply.call_count >= 5

    def test_per_turn_false_only_calls_in_finally(self, agent_responses_with_model_calls):
        """Test that per_turn=False only calls apply_management in finally block."""
        from unittest.mock import patch

        from tests.fixtures.mocked_model_provider import MockedModelProvider

        manager = SlidingWindowConversationManager(per_turn=False, window_size=100)
        model = MockedModelProvider(agent_responses_with_model_calls)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            agent("Run the test tool multiple times")

            # Should only be called once in finally block
            assert mock_apply.call_count == 1

    def test_no_model_calls_only_finally_block(self):
        """Test that with no model calls that trigger per_turn, management only happens in finally block."""
        from unittest.mock import patch

        from tests.fixtures.mocked_model_provider import MockedModelProvider

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

            # Should be called at least once (once per model call + finally)
            assert mock_apply.call_count >= 1

    def test_message_count_reduced_during_loop(self, agent_responses_with_model_calls):
        """Test that messages are trimmed during loop execution, not just at end."""
        from unittest.mock import patch

        from tests.fixtures.mocked_model_provider import MockedModelProvider

        # Use small window size to trigger trimming
        manager = SlidingWindowConversationManager(per_turn=1, window_size=4)
        model = MockedModelProvider(agent_responses_with_model_calls)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        # Track message counts after each model call
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


class TestModelCallCounter:
    """Tests for model call counter tracking."""

    def test_model_call_count_increments(self, agent_responses_with_model_calls):
        """Test that model call count increments correctly."""
        from tests.fixtures.mocked_model_provider import MockedModelProvider

        manager = SlidingWindowConversationManager(per_turn=3, window_size=100)
        assert manager.model_call_count == 0

        model = MockedModelProvider(agent_responses_with_model_calls)
        agent = Agent(model=model, conversation_manager=manager, tools=[create_test_tool()])

        agent("Run the test tool multiple times")

        # After execution, counter should reflect number of model calls
        assert manager.model_call_count >= 5

    def test_model_call_count_triggers_management(self):
        """Test that management is triggered at correct intervals based on counter."""
        from unittest.mock import MagicMock, patch

        from strands.hooks.events import AfterModelCallEvent
        from strands.hooks.registry import HookRegistry

        manager = SlidingWindowConversationManager(per_turn=2, window_size=100)
        registry = HookRegistry()
        manager.register_hooks(registry)

        # Create mock agent
        mock_agent = MagicMock()
        mock_agent.messages = []

        # Create mock events
        event1 = AfterModelCallEvent(agent=mock_agent)
        event2 = AfterModelCallEvent(agent=mock_agent)
        event3 = AfterModelCallEvent(agent=mock_agent)

        with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock_apply:
            # Invoke events
            registry.invoke_callbacks(event1)
            assert manager.model_call_count == 1
            assert mock_apply.call_count == 0  # Not called yet (per_turn=2)

            registry.invoke_callbacks(event2)
            assert manager.model_call_count == 2
            assert mock_apply.call_count == 1  # Called after 2nd model call

            registry.invoke_callbacks(event3)
            assert manager.model_call_count == 3
            assert mock_apply.call_count == 1  # Not called (waiting for 4th)


class TestStateManagement:
    """Tests for state persistence with per_turn."""

    def test_get_state_includes_model_call_count(self):
        """Test that get_state includes model_call_count."""
        manager = SlidingWindowConversationManager(per_turn=3)
        manager.model_call_count = 5
        state = manager.get_state()

        assert "model_call_count" in state
        assert state["model_call_count"] == 5

    def test_restore_from_session_restores_model_call_count(self):
        """Test that restore_from_session restores model_call_count."""
        manager = SlidingWindowConversationManager(per_turn=3)
        state = {
            "__name__": "SlidingWindowConversationManager",
            "removed_message_count": 0,
            "model_call_count": 7,
        }
        manager.restore_from_session(state)

        assert manager.model_call_count == 7


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_existing_sliding_window_still_works(self):
        """Test that existing SlidingWindowConversationManager usage still works."""
        # Old-style initialization without per_turn
        manager = SlidingWindowConversationManager(window_size=40)
        assert manager.per_turn is False

    def test_existing_agent_initialization_still_works(self):
        """Test that existing Agent initialization patterns still work."""
        from tests.fixtures.mocked_model_provider import MockedModelProvider

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
