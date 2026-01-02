"""Retry strategy implementations for handling model throttling and other retry scenarios.

This module provides hook-based retry strategies that can be configured on the Agent
to control retry behavior for model invocations. Retry strategies implement the
HookProvider protocol and register callbacks for AfterModelCallEvent to determine
when and how to retry failed model calls.
"""

import asyncio
import logging
from typing import Any

from ..types.exceptions import ModelThrottledException
from ..hooks.events import AfterInvocationEvent, AfterModelCallEvent
from ..hooks.registry import HookProvider, HookRegistry

logger = logging.getLogger(__name__)


class ModelRetryStrategy(HookProvider):
    """Default retry strategy for model throttling with exponential backoff.

    This strategy implements automatic retry logic for model throttling exceptions,
    using exponential backoff to handle rate limiting gracefully. It retries
    model calls when ModelThrottledException is raised, up to a configurable
    maximum number of attempts.

    The delay between retries starts at initial_delay and doubles after each
    retry, up to a maximum of max_delay. The strategy automatically resets
    its state after a successful model call.

    Example:
        ```python
        from strands import Agent
        from strands.hooks import ModelRetryStrategy

        # Use custom retry parameters
        retry_strategy = ModelRetryStrategy(
            max_attempts=3,
            initial_delay=2,
            max_delay=60
        )
        agent = Agent(retry_strategy=retry_strategy)
        ```

    Attributes:
        max_attempts: Maximum number of retry attempts before giving up.
        initial_delay: Initial delay in seconds before the first retry.
        max_delay: Maximum delay in seconds between retries.
        current_attempt: Current retry attempt counter (resets on success).
        current_delay: Current delay value for exponential backoff.
    """

    def __init__(
        self,
        max_attempts: int = 6,
        initial_delay: int = 4,
        max_delay: int = 240,
    ):
        """Initialize the retry strategy with the specified parameters.

        Args:
            max_attempts: Maximum number of retry attempts. Defaults to 6.
            initial_delay: Initial delay in seconds before retrying. Defaults to 4.
            max_delay: Maximum delay in seconds between retries. Defaults to 240 (4 minutes).
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.current_attempt = 0
        self.current_delay = initial_delay

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register callback for AfterModelCallEvent to handle retries.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        registry.add_callback(AfterModelCallEvent, self._handle_after_model_call)

    async def _handle_after_model_call(self, event: AfterModelCallEvent) -> None:
        """Handle model call completion and determine if retry is needed.

        This callback is invoked after each model call. If the call failed with
        a ModelThrottledException and we haven't exceeded max_attempts, it sets
        event.retry to True and sleeps for the current delay before returning.

        On successful calls, it resets the retry state to prepare for future calls.

        Args:
            event: The AfterModelCallEvent containing call results or exception.
        """
        # If already retrying, skip processing (another hook may have triggered retry)
        if event.retry:
            return

        # If model call succeeded, reset retry state
        if event.stop_response is not None:
            logger.debug(
                "stop_reason=<%s> | model call succeeded, resetting retry state",
                event.stop_response.stop_reason,
            )
            self._current_attempt = 0
            self._did_trigger_retry = False
            return

        # Check if we have an exception (and skip log if no exception)
        if event.exception is None:
            return

        # Only retry on ModelThrottledException
        if not isinstance(event.exception, ModelThrottledException):
            return

        # Increment attempt counter first
        self._current_attempt += 1

        # Check if we've exceeded max attempts
        if self._current_attempt >= self._max_attempts:
            logger.debug(
                "current_attempt=<%d>, max_attempts=<%d> | max retry attempts reached, not retrying",
                self._current_attempt,
                self._max_attempts,
            )
            self._did_trigger_retry = False
            return

        # Calculate delay for this attempt
        delay = self._calculate_delay()

        # Retry the model call
        logger.debug(
            "retry_delay_seconds=<%s>, max_attempts=<%s>, current_attempt=<%s> "
            "| throttling exception encountered | delaying before next retry",
            delay,
            self._max_attempts,
            self._current_attempt,
        )

        # Sleep for current delay
        await asyncio.sleep(delay)

        # Set retry flag and track that this strategy triggered it
        event.retry = True
        self._did_trigger_retry = True


class NoopRetryStrategy(HookProvider):
    """No-op retry strategy that disables automatic retries.

    This strategy can be used when you want to explicitly disable retry behavior
    and handle errors directly in your application code. It implements the
    HookProvider protocol but does not register any callbacks.

    Example:
        ```python
        from strands import Agent
        from strands.hooks import NoopRetryStrategy

        # Disable automatic retries
        agent = Agent(retry_strategy=NoopRetryStrategy())
        ```
    """

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        """Register hooks (no-op implementation).

        This method intentionally does nothing, as this strategy disables retries.

        Args:
            registry: The hook registry to register callbacks with.
            **kwargs: Additional keyword arguments for future extensibility.
        """
        # Intentionally empty - no callbacks to register
        pass
