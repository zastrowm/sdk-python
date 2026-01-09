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
