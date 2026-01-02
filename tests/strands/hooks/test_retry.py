"""Unit tests for retry strategy implementations."""

import asyncio
from unittest.mock import Mock

import pytest

from strands.hooks import AfterModelCallEvent, HookRegistry
from strands.hooks.retry import ModelRetryStrategy, NoopRetryStrategy
from strands.types.exceptions import ModelThrottledException


class TestModelRetryStrategy:
    """Tests for ModelRetryStrategy class."""

    def test_init_with_defaults(self):
        """Test ModelRetryStrategy initialization with default parameters."""
        strategy = ModelRetryStrategy()
        assert strategy.max_attempts == 6
        assert strategy.initial_delay == 4
        assert strategy.max_delay == 240
        assert strategy.current_attempt == 0
        assert strategy.current_delay == 4

    def test_init_with_custom_parameters(self):
        """Test ModelRetryStrategy initialization with custom parameters."""
        strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
        assert strategy.max_attempts == 3
        assert strategy.initial_delay == 2
        assert strategy.max_delay == 60
        assert strategy.current_attempt == 0
        assert strategy.current_delay == 2

    def test_register_hooks(self):
        """Test that ModelRetryStrategy registers AfterModelCallEvent callback."""
        strategy = ModelRetryStrategy()
        registry = HookRegistry()

        strategy.register_hooks(registry)

        # Verify callback was registered
        assert AfterModelCallEvent in registry._registered_callbacks
        assert len(registry._registered_callbacks[AfterModelCallEvent]) == 1

    @pytest.mark.asyncio
    async def test_retry_on_throttle_exception_first_attempt(self):
        """Test retry behavior on first ModelThrottledException."""
        strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
        mock_agent = Mock()

        event = AfterModelCallEvent(
            agent=mock_agent,
            exception=ModelThrottledException("Throttled"),
        )

        # Mock asyncio.sleep to avoid actual delays
        original_sleep = asyncio.sleep
        sleep_called_with = []

        async def mock_sleep(delay):
            sleep_called_with.append(delay)

        asyncio.sleep = mock_sleep
        try:
            await strategy._handle_after_model_call(event)

            # Should set retry to True
            assert event.retry is True
            # Should sleep for initial_delay
            assert sleep_called_with == [2]
            # Should increment attempt and double delay
            assert strategy.current_attempt == 1
            assert strategy.current_delay == 4
        finally:
            asyncio.sleep = original_sleep

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self):
        """Test exponential backoff calculation."""
        strategy = ModelRetryStrategy(max_attempts=5, initial_delay=2, max_delay=16)
        mock_agent = Mock()

        sleep_called_with = []

        async def mock_sleep(delay):
            sleep_called_with.append(delay)

        original_sleep = asyncio.sleep
        asyncio.sleep = mock_sleep

        try:
            # Simulate multiple retries
            for _ in range(4):
                event = AfterModelCallEvent(
                    agent=mock_agent,
                    exception=ModelThrottledException("Throttled"),
                )
                await strategy._handle_after_model_call(event)
                assert event.retry is True

            # Verify exponential backoff with max_delay cap
            # 2, 4, 8, 16 (capped), 16 (capped)
            assert sleep_called_with == [2, 4, 8, 16]
            # Delay should be capped at max_delay
            assert strategy.current_delay == 16
        finally:
            asyncio.sleep = original_sleep

    @pytest.mark.asyncio
    async def test_no_retry_after_max_attempts(self):
        """Test that retry is not set after reaching max_attempts."""
        strategy = ModelRetryStrategy(max_attempts=2, initial_delay=2, max_delay=60)
        mock_agent = Mock()

        async def mock_sleep(delay):
            pass

        original_sleep = asyncio.sleep
        asyncio.sleep = mock_sleep

        try:
            # First attempt
            event1 = AfterModelCallEvent(
                agent=mock_agent,
                exception=ModelThrottledException("Throttled"),
            )
            await strategy._handle_after_model_call(event1)
            assert event1.retry is True
            assert strategy.current_attempt == 1

            # Second attempt (at max_attempts)
            event2 = AfterModelCallEvent(
                agent=mock_agent,
                exception=ModelThrottledException("Throttled"),
            )
            await strategy._handle_after_model_call(event2)
            # Should NOT retry after reaching max_attempts
            assert event2.retry is False
            assert strategy.current_attempt == 2
        finally:
            asyncio.sleep = original_sleep

    @pytest.mark.asyncio
    async def test_no_retry_on_non_throttle_exception(self):
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
        assert strategy.current_attempt == 0

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self):
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
    async def test_reset_on_success(self):
        """Test that strategy resets attempt counter on successful call."""
        strategy = ModelRetryStrategy(max_attempts=3, initial_delay=2, max_delay=60)
        mock_agent = Mock()

        async def mock_sleep(delay):
            pass

        original_sleep = asyncio.sleep
        asyncio.sleep = mock_sleep

        try:
            # First failure
            event1 = AfterModelCallEvent(
                agent=mock_agent,
                exception=ModelThrottledException("Throttled"),
            )
            await strategy._handle_after_model_call(event1)
            assert event1.retry is True
            assert strategy.current_attempt == 1

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
            assert strategy.current_attempt == 0
            assert strategy.current_delay == 2
        finally:
            asyncio.sleep = original_sleep


class TestNoopRetryStrategy:
    """Tests for NoopRetryStrategy class."""

    def test_register_hooks_does_nothing(self):
        """Test that NoopRetryStrategy does not register any callbacks."""
        strategy = NoopRetryStrategy()
        registry = HookRegistry()

        strategy.register_hooks(registry)

        # Verify no callbacks were registered
        assert len(registry._registered_callbacks) == 0

    @pytest.mark.asyncio
    async def test_no_retry_on_throttle_exception(self):
        """Test that NoopRetryStrategy does not retry on throttle exceptions."""
        # This test verifies that with NoopRetryStrategy, the event.retry
        # remains False even on throttling exceptions
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
