import pytest

from strands.agent.agent import Agent
from strands.agent.conversation_manager.null_conversation_manager import NullConversationManager
from strands.agent.conversation_manager.sliding_window_conversation_manager import SlidingWindowConversationManager
from strands.hooks.events import BeforeModelCallEvent
from strands.hooks.registry import HookRegistry
from strands.types.exceptions import ContextWindowOverflowException
from tests.fixtures.mocked_model_provider import MockedModelProvider


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


# ==============================================================================
# Per-Turn Management Tests
# ==============================================================================


@pytest.mark.parametrize(
    "per_turn_value,is_valid",
    [
        (False, True),
        (True, True),
        (1, True),
        (3, True),
        (0, False),
        (-1, False),
    ],
)
def test_per_turn_parameter_validation(per_turn_value, is_valid):
    """Test per_turn parameter validation for various values."""
    if is_valid:
        manager = SlidingWindowConversationManager(per_turn=per_turn_value)
        assert manager.per_turn == per_turn_value
    else:
        with pytest.raises(ValueError):
            SlidingWindowConversationManager(per_turn=per_turn_value)


@pytest.mark.parametrize(
    "per_turn,num_responses,expected_min_calls",
    [
        (False, 1, 1),  # Only finally block
        (True, 1, 2),  # Before model call + finally
        (2, 1, 1),  # Not reached, only finally
        (3, 1, 1),  # Not reached, only finally
    ],
)
def test_per_turn_management_frequency(per_turn, num_responses, expected_min_calls):
    """Test apply_management call frequency with different per_turn values."""
    from unittest.mock import patch

    manager = SlidingWindowConversationManager(per_turn=per_turn, window_size=100)
    responses = [{"role": "assistant", "content": [{"text": f"Response {i}"}]} for i in range(num_responses)]
    model = MockedModelProvider(responses)
    agent = Agent(model=model, conversation_manager=manager)

    with patch.object(manager, "apply_management", wraps=manager.apply_management) as mock:
        agent("Test")
        assert mock.call_count >= expected_min_calls


def test_per_turn_dynamic_change():
    """Test that per_turn can be changed dynamically during execution."""
    from unittest.mock import MagicMock, patch

    manager = SlidingWindowConversationManager(per_turn=False)
    registry = HookRegistry()
    manager.register_hooks(registry)

    mock_agent = MagicMock()
    mock_agent.messages = []
    event = BeforeModelCallEvent(agent=mock_agent)

    # Initially disabled
    with patch.object(manager, "apply_management") as mock_apply:
        registry.invoke_callbacks(event)
        assert mock_apply.call_count == 0

    # Enable dynamically
    manager.per_turn = True
    with patch.object(manager, "apply_management") as mock_apply:
        registry.invoke_callbacks(event)
        assert mock_apply.call_count == 1


def test_per_turn_state_persistence():
    """Test that model_call_count is persisted and restored from state."""
    manager = SlidingWindowConversationManager(per_turn=3)
    manager.model_call_count = 7

    state = manager.get_state()
    assert state["model_call_count"] == 7

    new_manager = SlidingWindowConversationManager(per_turn=3)
    new_manager.restore_from_session(state)
    assert new_manager.model_call_count == 7
