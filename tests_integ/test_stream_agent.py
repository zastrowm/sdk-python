"""
Test script for Strands' custom callback handler functionality.
Demonstrates different patterns of callback handling and processing.
"""

import logging

from strands import Agent

logging.getLogger("strands").setLevel(logging.DEBUG)
logging.basicConfig(format="%(levelname)s | %(name)s | %(message)s", handlers=[logging.StreamHandler()])


class ToolCountingCallbackHandler:
    def __init__(self):
        self.tool_count = 0
        self.message_count = 0

    def callback_handler(self, **kwargs) -> None:
        """
        Custom callback handler that processes and displays different types of events.

        Args:
            **kwargs: Callback event data including:
                - data: Regular output
                - complete: Completion status
                - message: Message processing
                - current_tool_use: Tool execution
        """
        # Extract event data
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        message = kwargs.get("message", {})
        current_tool_use = kwargs.get("current_tool_use", {})

        # Handle regular data output
        if data:
            print(f"🔄 Data: {data}")

        # Handle tool execution events
        if current_tool_use:
            self.tool_count += 1
            tool_name = current_tool_use.get("name", "")
            tool_input = current_tool_use.get("input", {})
            print(f"🛠️ Tool Execution #{self.tool_count}\nTool: {tool_name}\nInput: {tool_input}")

        # Handle message processing
        if message:
            self.message_count += 1
            print(f"📝 Message #{self.message_count}")

        # Handle completion
        if complete:
            self.console.print("✨ Callback Complete", style="bold green")


def test_basic_interaction():
    """Test basic AGI interaction with custom callback handler."""
    print("\nTesting Basic Interaction")

    # Initialize agent with custom handler
    agent = Agent(
        callback_handler=ToolCountingCallbackHandler().callback_handler,
        load_tools_from_directory=False,
    )

    # Simple prompt to test callbacking
    agent("Tell me a short joke from your general knowledge")

    print("\nBasic Interaction Complete")


# ============================================================================
# Concurrency Exception Integration Tests
# ============================================================================


def test_concurrent_invocations_with_threading():
    """Integration test: Concurrent agent invocations with real threading."""
    import threading
    from strands.types.exceptions import ConcurrencyException
    from tests.fixtures.mocked_model_provider import MockedModelProvider

    model = MockedModelProvider(
        [
            {"role": "assistant", "content": [{"text": "response1"}]},
            {"role": "assistant", "content": [{"text": "response2"}]},
        ]
    )
    agent = Agent(model=model, callback_handler=None)

    results = []
    errors = []
    lock = threading.Lock()

    def invoke():
        try:
            result = agent("test prompt")
            with lock:
                results.append(result)
        except ConcurrencyException as e:
            with lock:
                errors.append(e)

    print("\nTesting concurrent invocations with threading")

    t1 = threading.Thread(target=invoke)
    t2 = threading.Thread(target=invoke)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Verify one succeeded and one raised exception
    print(f"Successful invocations: {len(results)}")
    print(f"Raised ConcurrencyExceptions: {len(errors)}")

    assert len(results) == 1, f"Expected 1 success, got {len(results)}"
    assert len(errors) == 1, f"Expected 1 error, got {len(errors)}"
    assert "concurrent" in str(errors[0]).lower() and "invocation" in str(errors[0]).lower()

    print("Concurrent invocation test passed")


def test_retry_scenario_with_timeout():
    """Integration test: Simulate client timeout retry scenario."""
    import threading
    import time
    from strands.types.exceptions import ConcurrencyException
    from tests.fixtures.mocked_model_provider import MockedModelProvider

    # Create a slow-responding model
    class SlowMockedModel(MockedModelProvider):
        async def stream(self, messages, tool_specs=None, system_prompt=None, tool_choice=None, **kwargs):
            # Simulate slow response
            import asyncio

            await asyncio.sleep(0.2)
            async for event in super().stream(messages, tool_specs, system_prompt, tool_choice, **kwargs):
                yield event

    model = SlowMockedModel(
        [
            {"role": "assistant", "content": [{"text": "slow response"}]},
            {"role": "assistant", "content": [{"text": "retry response"}]},
        ]
    )
    agent = Agent(model=model, callback_handler=None)

    first_result = []
    retry_error = []
    lock = threading.Lock()

    def first_request():
        try:
            result = agent("process this request")
            with lock:
                first_result.append(result)
        except Exception as e:
            with lock:
                first_result.append(e)

    def retry_request():
        # Wait a bit before retrying (simulating client timeout retry)
        time.sleep(0.1)
        try:
            result = agent("process this request")  # Same request, retry
            with lock:
                retry_error.append(f"Unexpected success: {result}")
        except ConcurrencyException as e:
            with lock:
                retry_error.append(e)

    print("\nTesting retry scenario with timeout")

    t1 = threading.Thread(target=first_request)
    t2 = threading.Thread(target=retry_request)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # First request should succeed
    assert len(first_result) == 1
    print(f"First request: {'Success' if hasattr(first_result[0], 'message') else 'Failed'}")

    # Retry should raise ConcurrencyException
    assert len(retry_error) == 1
    assert isinstance(retry_error[0], ConcurrencyException)
    print(f"Retry raised: {type(retry_error[0]).__name__}")

    print("Retry scenario test passed")
