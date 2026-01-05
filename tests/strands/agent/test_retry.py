"""Unit tests for retry strategy implementations."""

from unittest.mock import Mock

import pytest

from strands.hooks import AfterModelCallEvent, HookRegistry
from strands.agent.retry import ModelRetryStrategy, NoopRetryStrategy
from strands.types.exceptions import ModelThrottledException


# ModelRetryStrategy Tests


def test_model_retry_strategy_init_with_defaults():
    """Test ModelRetryStrategy initialization with default parameters."""
    strategy = ModelRetryStrategy()
    assert strategy._max_attempts == 6
    assert strategy._initial_delay == 4
    assert strategy._max_delay == 240
    assert strategy._current_attempt == 0
    assert strategy._calculate_delay() == 4


def test_model_retry_strategy_init_with_custom_parameters():
    """Test ModelRetryStrategy initialization with custom parameters."""
    strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
    assert strategy._max_attempts == 3
    assert strategy._initial_delay == 2
    assert strategy._max_delay == 60
    assert strategy._current_attempt == 0
    assert strategy._calculate_delay() == 2


def test_model_retry_strategy_register_hooks():
    """Test that ModelRetryStrategy registers AfterModelCallEvent callback."""
    strategy = ModelRetryStrategy()
    registry = HookRegistry()

    strategy.register_hooks(registry)

    # Verify callback was registered
    assert AfterModelCallEvent in registry._registered_callbacks
    assert len(registry._registered_callbacks[AfterModelCallEvent]) == 1


@pytest.mark.asyncio
async def test_model_retry_strategy_retry_on_throttle_exception_first_attempt(mock_sleep):
    """Test retry behavior on first ModelThrottledException."""
    strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
    mock_agent = Mock()

    event = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )

    await strategy._handle_after_model_call(event)

    # Should set retry to True
    assert event.retry is True
    # Should sleep for initial_delay
    assert mock_sleep.sleep_calls == [2]
    # Should increment attempt
    assert strategy._current_attempt == 1
    assert strategy._calculate_delay() == 4


@pytest.mark.asyncio
async def test_model_retry_strategy_exponential_backoff(mock_sleep):
    """Test exponential backoff calculation."""
    strategy = ModelRetryStrategy(max_attempts=5, initial_delay=2, max_delay=16)
    mock_agent = Mock()

    # Simulate multiple retries
    for _ in range(4):
        event = AfterModelCallEvent(
            agent=mock_agent,
            exception=ModelThrottledException("Throttled"),
        )
        await strategy._handle_after_model_call(event)
        assert event.retry is True

    # Verify exponential backoff with max_delay cap
    # 2, 4, 8, 16 (capped)
    assert mock_sleep.sleep_calls == [2, 4, 8, 16]
    # Delay should be capped at max_delay
    assert strategy._calculate_delay() == 16


@pytest.mark.asyncio
async def test_model_retry_strategy_no_retry_after_max_attempts(mock_sleep):
    """Test that retry is not set after reaching max_attempts."""
    strategy = ModelRetryStrategy(max_attempts=2, initial_delay=2, max_delay=60)
    mock_agent = Mock()

    # First attempt
    event1 = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )
    await strategy._handle_after_model_call(event1)
    assert event1.retry is True
    assert strategy._current_attempt == 1

    # Second attempt (at max_attempts)
    event2 = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )
    await strategy._handle_after_model_call(event2)
    # Should NOT retry after reaching max_attempts
    assert event2.retry is False
    assert strategy._current_attempt == 2


@pytest.mark.asyncio
async def test_model_retry_strategy_no_retry_on_non_throttle_exception():
    """Test that retry is not set for non-throttling exceptions."""
    strategy = ModelRetryStrategy()
    mock_agent = Mock()

    event = AfterModelCallEvent(
        agent=mock_agent,
        exception=ValueError("Some other error"),
    )

    await strategy._handle_after_model_call(event)

    # Should not retry on non-throttling exceptions
    assert event.retry is False
    assert strategy._current_attempt == 0


@pytest.mark.asyncio
async def test_model_retry_strategy_no_retry_on_success():
    """Test that retry is not set when model call succeeds."""
    strategy = ModelRetryStrategy()
    mock_agent = Mock()

    event = AfterModelCallEvent(
        agent=mock_agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message={"role": "assistant", "content": [{"text": "Success"}]},
            stop_reason="end_turn",
        ),
    )

    await strategy._handle_after_model_call(event)

    # Should not retry on success
    assert event.retry is False


@pytest.mark.asyncio
async def test_model_retry_strategy_reset_on_success(mock_sleep):
    """Test that strategy resets attempt counter on successful call."""
    strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
    mock_agent = Mock()

    # First failure
    event1 = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )
    await strategy._handle_after_model_call(event1)
    assert event1.retry is True
    assert strategy._current_attempt == 1

    # Success - should reset
    event2 = AfterModelCallEvent(
        agent=mock_agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message={"role": "assistant", "content": [{"text": "Success"}]},
            stop_reason="end_turn",
        ),
    )
    await strategy._handle_after_model_call(event2)
    assert event2.retry is False
    # Should reset to initial state
    assert strategy._current_attempt == 0
    assert strategy._calculate_delay() == 2


# NoopRetryStrategy Tests


def test_noop_retry_strategy_register_hooks_does_nothing():
    """Test that NoopRetryStrategy does not register any callbacks."""
    strategy = NoopRetryStrategy()
    registry = HookRegistry()

    strategy.register_hooks(registry)

    # Verify no callbacks were registered
    assert len(registry._registered_callbacks) == 0


@pytest.mark.asyncio
async def test_noop_retry_strategy_no_retry_on_throttle_exception():
    """Test that NoopRetryStrategy does not retry on throttle exceptions."""
    strategy = NoopRetryStrategy()
    registry = HookRegistry()
    strategy.register_hooks(registry)

    mock_agent = Mock()
    event = AfterModelCallEvent(
        agent=mock_agent,
        exception=ModelThrottledException("Throttled"),
    )

    # Invoke callbacks (should be none registered)
    await registry.invoke_callbacks_async(event)

    # event.retry should still be False (default)
    assert event.retry is False

