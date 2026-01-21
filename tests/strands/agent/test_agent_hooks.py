from unittest.mock import ANY, AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    AfterToolCallEvent,
    AfterToolsEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    BeforeToolsEvent,
    MessageAddedEvent,
)
from strands.types.content import Messages
from strands.types.exceptions import ModelThrottledException
from strands.types.tools import ToolResult, ToolUse
from tests.fixtures.mock_hook_provider import MockHookProvider
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def hook_provider():
    return MockHookProvider(
        [
            AgentInitializedEvent,
            BeforeInvocationEvent,
            AfterInvocationEvent,
            AfterToolCallEvent,
            BeforeToolCallEvent,
            BeforeModelCallEvent,
            AfterModelCallEvent,
            MessageAddedEvent,
        ]
    )


@pytest.fixture
def agent_tool():
    @strands.tools.tool(name="tool_decorated")
    def reverse(random_string: str) -> str:
        return random_string[::-1]

    return reverse


@pytest.fixture
def tool_use(agent_tool):
    return {"name": agent_tool.tool_name, "toolUseId": "123", "input": {"random_string": "I invoked a tool!"}}


@pytest.fixture
def mock_model(tool_use):
    agent_messages: Messages = [
        {
            "role": "assistant",
            "content": [{"toolUse": tool_use}],
        },
        {"role": "assistant", "content": [{"text": "I invoked a tool!"}]},
    ]
    return MockedModelProvider(agent_messages)


@pytest.fixture
def agent(
    mock_model,
    hook_provider,
    agent_tool,
):
    agent = Agent(
        model=mock_model,
        system_prompt="You are a helpful assistant.",
        callback_handler=None,
        tools=[agent_tool],
    )

    hooks = agent.hooks
    hooks.add_hook(hook_provider)

    def assert_message_is_last_message_added(event: MessageAddedEvent):
        assert event.agent.messages[-1] == event.message

    hooks.add_callback(MessageAddedEvent, assert_message_is_last_message_added)

    return agent


@pytest.fixture
def tools_config(agent):
    return agent.tool_config["tools"]


@pytest.fixture
def user():
    class User(BaseModel):
        name: str
        age: int

    return User(name="Jane Doe", age=30)


@pytest.fixture
def mock_sleep():
    with patch.object(strands.event_loop.event_loop.asyncio, "sleep", new_callable=AsyncMock) as mock:
        yield mock


def test_agent__init__hooks():
    """Verify that the AgentInitializedEvent is emitted on Agent construction."""
    hook_provider = MockHookProvider(event_types=[AgentInitializedEvent])
    agent = Agent(hooks=[hook_provider])

    length, events = hook_provider.get_events()

    assert length == 1

    assert next(events) == AgentInitializedEvent(agent=agent)


def test_agent_tool_call(agent, hook_provider, agent_tool):
    agent.tool.tool_decorated(random_string="a string")

    length, events = hook_provider.get_events()

    tool_use: ToolUse = {"input": {"random_string": "a string"}, "name": "tool_decorated", "toolUseId": ANY}
    result: ToolResult = {"content": [{"text": "gnirts a"}], "status": "success", "toolUseId": ANY}

    assert length == 6

    assert next(events) == BeforeToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
    )
    assert next(events) == AfterToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
        result=result,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[0])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert len(agent.messages) == 4


def test_agent__call__hooks(agent, hook_provider, agent_tool, mock_model, tool_use):
    """Verify that the correct hook events are emitted as part of __call__."""

    result = agent("test message")

    length, events = hook_provider.get_events()

    assert length == 12

    assert next(events) == BeforeInvocationEvent(agent=agent)
    assert next(events) == MessageAddedEvent(
        agent=agent,
        message=agent.messages[0],
    )
    assert next(events) == BeforeModelCallEvent(agent=agent)
    assert next(events) == AfterModelCallEvent(
        agent=agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message={
                "content": [{"toolUse": tool_use}],
                "role": "assistant",
            },
            stop_reason="tool_use",
        ),
        exception=None,
    )

    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
    assert next(events) == BeforeToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
    )
    assert next(events) == AfterToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
        result={"content": [{"text": "!loot a dekovni I"}], "status": "success", "toolUseId": "123"},
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == BeforeModelCallEvent(agent=agent)
    assert next(events) == AfterModelCallEvent(
        agent=agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message=mock_model.agent_responses[1],
            stop_reason="end_turn",
        ),
        exception=None,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert next(events) == AfterInvocationEvent(agent=agent, result=result)

    assert len(agent.messages) == 4


@pytest.mark.asyncio
async def test_agent_stream_async_hooks(agent, hook_provider, agent_tool, mock_model, tool_use, agenerator):
    """Verify that the correct hook events are emitted as part of stream_async."""
    iterator = agent.stream_async("test message")
    await anext(iterator)
    assert hook_provider.events_received == [BeforeInvocationEvent(agent=agent)]

    # iterate the rest
    result = None
    async for item in iterator:
        if "result" in item:
            result = item["result"]

    length, events = hook_provider.get_events()

    assert length == 12

    assert next(events) == BeforeInvocationEvent(agent=agent)
    assert next(events) == MessageAddedEvent(
        agent=agent,
        message=agent.messages[0],
    )
    assert next(events) == BeforeModelCallEvent(agent=agent)
    assert next(events) == AfterModelCallEvent(
        agent=agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message={
                "content": [{"toolUse": tool_use}],
                "role": "assistant",
            },
            stop_reason="tool_use",
        ),
        exception=None,
    )

    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[1])
    assert next(events) == BeforeToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
    )
    assert next(events) == AfterToolCallEvent(
        agent=agent,
        selected_tool=agent_tool,
        tool_use=tool_use,
        invocation_state=ANY,
        result={"content": [{"text": "!loot a dekovni I"}], "status": "success", "toolUseId": "123"},
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[2])
    assert next(events) == BeforeModelCallEvent(agent=agent)
    assert next(events) == AfterModelCallEvent(
        agent=agent,
        stop_response=AfterModelCallEvent.ModelStopResponse(
            message=mock_model.agent_responses[1],
            stop_reason="end_turn",
        ),
        exception=None,
    )
    assert next(events) == MessageAddedEvent(agent=agent, message=agent.messages[3])

    assert next(events) == AfterInvocationEvent(agent=agent, result=result)

    assert len(agent.messages) == 4


def test_agent_structured_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output."""

    agent.model.structured_output = Mock(return_value=agenerator([{"output": user}]))
    agent.structured_output(type(user), "example prompt")

    length, events = hook_provider.get_events()

    assert length == 2

    assert next(events) == BeforeInvocationEvent(agent=agent)
    assert next(events) == AfterInvocationEvent(agent=agent)

    assert len(agent.messages) == 0  # no new messages added


@pytest.mark.asyncio
async def test_agent_structured_async_output_hooks(agent, hook_provider, user, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output_async."""

    agent.model.structured_output = Mock(return_value=agenerator([{"output": user}]))
    await agent.structured_output_async(type(user), "example prompt")

    length, events = hook_provider.get_events()

    assert length == 2

    assert next(events) == BeforeInvocationEvent(agent=agent)
    assert next(events) == AfterInvocationEvent(agent=agent)

    assert len(agent.messages) == 0  # no new messages added


@pytest.mark.asyncio
async def test_hook_retry_on_successful_call():
    """Test that hooks can retry even on successful model calls based on response content."""

    mock_provider = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"text": "Short"}],
            },
            {
                "role": "assistant",
                "content": [{"text": "This is a much longer and more detailed response"}],
            },
        ]
    )

    # Hook that retries if response is too short
    class MinLengthRetryHook:
        def __init__(self, min_length=10):
            self.min_length = min_length
            self.call_count = 0

        def register_hooks(self, registry):
            registry.add_callback(strands.hooks.AfterModelCallEvent, self.handle_after_model_call)

        async def handle_after_model_call(self, event):
            self.call_count += 1

            # Check successful responses for minimum length
            if event.stop_response:
                message = event.stop_response.message
                text_content = "".join(block.get("text", "") for block in message.get("content", []))

                if len(text_content) < self.min_length:
                    event.retry = True

    retry_hook = MinLengthRetryHook(min_length=10)
    agent = Agent(model=mock_provider, hooks=[retry_hook])

    result = agent("Generate a response")

    # Verify hook was called twice (once for short response, once for long)
    assert retry_hook.call_count == 2

    # Verify final result is the longer response
    assert result.message["content"][0]["text"] == "This is a much longer and more detailed response"


@pytest.mark.asyncio
async def test_hook_retry_on_exception_basic(alist, mock_sleep):
    """Test that hooks can retry model calls on exceptions."""

    class CustomException(Exception):
        pass

    model = MagicMock()
    model.stream.side_effect = [
        CustomException("First attempt fails"),
        MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Success after retry"}],
                },
            ]
        ).stream([]),
    ]

    # Hook that enables retry on CustomException
    class RetryHook:
        def __init__(self):
            self.after_model_call_count = 0

        def register_hooks(self, registry):
            registry.add_callback(strands.hooks.AfterModelCallEvent, self.handle_after_model_call)

        async def handle_after_model_call(self, event):
            self.after_model_call_count += 1
            if event.exception and isinstance(event.exception, CustomException):
                event.retry = True

    retry_hook = RetryHook()
    agent = Agent(model=model, hooks=[retry_hook])

    result = agent("Test retry")

    # Verify the hook was called twice (once for failure, once for success)
    assert retry_hook.after_model_call_count == 2
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Success after retry"


@pytest.mark.asyncio
async def test_hook_retry_not_set_on_success(alist):
    """Test that model is not retried when hook doesn't set retry_model on success."""
    mock_provider = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [{"text": "First successful response"}],
            },
        ]
    )

    # Hook that tries to set retry_model=True even on success
    class NoRetryHook:
        def __init__(self):
            self.call_count = 0

        def register_hooks(self, registry):
            registry.add_callback(strands.hooks.AfterModelCallEvent, self.handle_after_model_call)

        async def handle_after_model_call(self, event):
            self.call_count += 1
            # Try to set retry even on success
            # Don't set retry_model (leave it as False)

    retry_hook = NoRetryHook()
    agent = Agent(model=mock_provider, hooks=[retry_hook])

    result = agent("Test no retry when not set")

    # Should only be called once since retry_model was not set
    assert retry_hook.call_count == 1
    assert result.message["content"][0]["text"] == "First successful response"


@pytest.mark.asyncio
async def test_hook_retry_with_limit(alist, mock_sleep):
    """Test that hooks can control retry limits."""

    class CustomException(Exception):
        pass

    model = MagicMock()
    model.stream.side_effect = [
        CustomException("Attempt 1 fails"),
        CustomException("Attempt 2 fails"),
        CustomException("Attempt 3 fails"),
    ]

    # Hook that allows max 2 retries
    class LimitedRetryHook:
        def __init__(self, max_retries=2):
            self.max_retries = max_retries
            self.retry_count = 0
            self.call_count = 0

        def register_hooks(self, registry):
            registry.add_callback(strands.hooks.AfterModelCallEvent, self.handle_after_model_call)

        async def handle_after_model_call(self, event):
            self.call_count += 1
            if event.exception and isinstance(event.exception, CustomException):
                if self.retry_count < self.max_retries:
                    self.retry_count += 1
                    event.retry = True
                # else: let exception propagate

    retry_hook = LimitedRetryHook(max_retries=2)
    agent = Agent(model=model, hooks=[retry_hook])

    with pytest.raises(CustomException, match="Attempt 3 fails"):
        await agent("Test limited retries")

    # Should be called 3 times: initial + 2 retries
    assert retry_hook.call_count == 3


# Tests for BeforeToolsEvent and AfterToolsEvent


def test_before_tools_event_triggered(agent, hook_provider, agent_tool, tool_use):
    """Verify that BeforeToolsEvent is triggered before tool batch execution."""
    # Add batch event tracking
    batch_hook_provider = MockHookProvider([BeforeToolsEvent, AfterToolsEvent])
    agent.hooks.add_hook(batch_hook_provider)

    result = agent("test message")

    # Check that BeforeToolsEvent was triggered
    batch_length, batch_events = batch_hook_provider.get_events()
    assert batch_length == 2  # BeforeToolsEvent and AfterToolsEvent

    before_event = next(batch_events)
    assert isinstance(before_event, BeforeToolsEvent)
    assert before_event.agent == agent
    assert len(before_event.tool_uses) == 1
    assert before_event.tool_uses[0]["name"] == "tool_decorated"
    assert "toolUse" in before_event.message["content"][0]


def test_after_tools_event_triggered(agent, hook_provider, agent_tool, tool_use):
    """Verify that AfterToolsEvent is triggered after all tools complete."""
    # Add batch event tracking
    batch_hook_provider = MockHookProvider([BeforeToolsEvent, AfterToolsEvent])
    agent.hooks.add_hook(batch_hook_provider)

    result = agent("test message")

    # Check that AfterToolsEvent was triggered
    batch_length, batch_events = batch_hook_provider.get_events()
    assert batch_length == 2

    before_event = next(batch_events)
    after_event = next(batch_events)
    
    assert isinstance(after_event, AfterToolsEvent)
    assert after_event.agent == agent
    assert len(after_event.tool_uses) == 1
    assert after_event.tool_uses[0]["name"] == "tool_decorated"
    assert "toolUse" in after_event.message["content"][0]


def test_after_tools_event_reverse_ordering():
    """Verify that AfterToolsEvent uses reverse callback ordering."""
    execution_order = []

    class OrderTrackingHook1:
        def register_hooks(self, registry):
            registry.add_callback(AfterToolsEvent, lambda event: execution_order.append("hook1"))

    class OrderTrackingHook2:
        def register_hooks(self, registry):
            registry.add_callback(AfterToolsEvent, lambda event: execution_order.append("hook2"))

    @strands.tools.tool
    def sample_tool(x: int) -> int:
        return x * 2

    tool_use = {"name": "sample_tool", "toolUseId": "123", "input": {"x": 5}}
    agent_messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": tool_use}]},
        {"role": "assistant", "content": [{"text": "Done"}]},
    ]
    model = MockedModelProvider(agent_messages)

    agent = Agent(
        model=model,
        tools=[sample_tool],
        hooks=[OrderTrackingHook1(), OrderTrackingHook2()],
    )

    agent("test")

    # AfterToolsEvent should execute in reverse order: hook2 before hook1
    assert execution_order == ["hook2", "hook1"]


def test_before_tools_event_with_multiple_tools():
    """Verify that BeforeToolsEvent contains all tools in batch."""
    batch_hook_provider = MockHookProvider([BeforeToolsEvent, AfterToolsEvent])

    @strands.tools.tool
    def tool1(x: int) -> int:
        return x + 1

    @strands.tools.tool
    def tool2(y: int) -> int:
        return y * 2

    tool_use_1 = {"name": "tool1", "toolUseId": "123", "input": {"x": 5}}
    tool_use_2 = {"name": "tool2", "toolUseId": "456", "input": {"y": 10}}
    
    agent_messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": tool_use_1}, {"toolUse": tool_use_2}]},
        {"role": "assistant", "content": [{"text": "Done"}]},
    ]
    model = MockedModelProvider(agent_messages)

    agent = Agent(
        model=model,
        tools=[tool1, tool2],
        hooks=[batch_hook_provider],
    )

    agent("test")

    batch_length, batch_events = batch_hook_provider.get_events()
    before_event = next(batch_events)
    
    assert isinstance(before_event, BeforeToolsEvent)
    assert len(before_event.tool_uses) == 2
    assert before_event.tool_uses[0]["name"] == "tool1"
    assert before_event.tool_uses[1]["name"] == "tool2"


def test_batch_events_not_triggered_without_tools():
    """Verify that batch events are not triggered when no tools are present."""
    batch_hook_provider = MockHookProvider([BeforeToolsEvent, AfterToolsEvent])

    # Response with no tool uses
    agent_messages: Messages = [
        {"role": "assistant", "content": [{"text": "No tools used"}]},
    ]
    model = MockedModelProvider(agent_messages)

    agent = Agent(
        model=model,
        hooks=[batch_hook_provider],
    )

    agent("test")

    # No batch events should be triggered
    batch_length, _ = batch_hook_provider.get_events()
    assert batch_length == 0


def test_before_tools_event_interrupt():
    """Verify that BeforeToolsEvent interrupt stops batch execution."""
    batch_hook_provider = MockHookProvider([BeforeToolsEvent, AfterToolsEvent])
    tool_hook_provider = MockHookProvider([BeforeToolCallEvent, AfterToolCallEvent])

    class InterruptHook:
        def register_hooks(self, registry):
            registry.add_callback(BeforeToolsEvent, self.interrupt_batch)

        def interrupt_batch(self, event: BeforeToolsEvent):
            # Interrupt without providing response
            event.interrupt("batch-approval", reason="Need approval")

    @strands.tools.tool
    def sample_tool(x: int) -> int:
        return x * 2

    tool_use = {"name": "sample_tool", "toolUseId": "123", "input": {"x": 5}}
    agent_messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": tool_use}]},
    ]
    model = MockedModelProvider(agent_messages)

    agent = Agent(
        model=model,
        tools=[sample_tool],
        hooks=[InterruptHook(), batch_hook_provider, tool_hook_provider],
    )

    result = agent("test")

    # Agent should stop with interrupt
    assert result.stop_reason == "interrupt"
    assert len(result.interrupts) == 1
    assert result.interrupts[0].name == "batch-approval"

    # Both BeforeToolsEvent and AfterToolsEvent should be triggered
    batch_length, batch_events = batch_hook_provider.get_events()
    assert batch_length == 2  # BeforeToolsEvent and AfterToolsEvent
    event1 = next(batch_events)
    event2 = next(batch_events)
    assert isinstance(event1, BeforeToolsEvent)
    assert isinstance(event2, AfterToolsEvent)

    # No individual tool events should be triggered (tools didn't execute)
    tool_length, _ = tool_hook_provider.get_events()
    assert tool_length == 0


@pytest.mark.asyncio
async def test_before_tools_event_interrupt_async():
    """Verify that BeforeToolsEvent interrupt works in async context."""
    batch_hook_provider = MockHookProvider([BeforeToolsEvent, AfterToolsEvent])

    class AsyncInterruptHook:
        def register_hooks(self, registry):
            registry.add_callback(BeforeToolsEvent, self.interrupt_batch)

        async def interrupt_batch(self, event: BeforeToolsEvent):
            event.interrupt("async-batch-approval", reason="Async approval needed")

    @strands.tools.tool
    def sample_tool(x: int) -> int:
        return x * 2

    tool_use = {"name": "sample_tool", "toolUseId": "123", "input": {"x": 5}}
    agent_messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": tool_use}]},
    ]
    model = MockedModelProvider(agent_messages)

    agent = Agent(
        model=model,
        tools=[sample_tool],
        hooks=[AsyncInterruptHook(), batch_hook_provider],
    )

    # Call agent synchronously but the hook is async
    result = agent("test")

    assert result.stop_reason == "interrupt"
    assert len(result.interrupts) == 1
    assert result.interrupts[0].name == "async-batch-approval"
    
    # Both BeforeToolsEvent and AfterToolsEvent should be triggered
    batch_length, _ = batch_hook_provider.get_events()
    assert batch_length == 2


def test_batch_events_with_tool_events():
    """Verify that batch events and per-tool events are triggered in correct order."""
    all_hook_provider = MockHookProvider([
        BeforeToolsEvent,
        AfterToolsEvent,
        BeforeToolCallEvent,
        AfterToolCallEvent,
    ])

    @strands.tools.tool
    def sample_tool(x: int) -> int:
        return x * 2

    tool_use = {"name": "sample_tool", "toolUseId": "123", "input": {"x": 5}}
    agent_messages: Messages = [
        {"role": "assistant", "content": [{"toolUse": tool_use}]},
        {"role": "assistant", "content": [{"text": "Done"}]},
    ]
    model = MockedModelProvider(agent_messages)

    agent = Agent(
        model=model,
        tools=[sample_tool],
        hooks=[all_hook_provider],
    )

    agent("test")

    event_length, events = all_hook_provider.get_events()
    assert event_length == 4

    # Expected order: BeforeToolsEvent, BeforeToolCallEvent, AfterToolCallEvent, AfterToolsEvent
    event_list = list(events)
    assert isinstance(event_list[0], BeforeToolsEvent)
    assert isinstance(event_list[1], BeforeToolCallEvent)
    assert isinstance(event_list[2], AfterToolCallEvent)
    assert isinstance(event_list[3], AfterToolsEvent)


@pytest.mark.asyncio
async def test_hook_retry_multiple_hooks(alist, mock_sleep):
    """Test that multiple hooks can modify retry_model and last one wins."""

    class CustomException(Exception):
        pass

    model = MagicMock()
    model.stream.side_effect = [
        CustomException("First attempt fails"),
        MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Success"}],
                },
            ]
        ).stream([]),
    ]

    async def retry_enabler(event: AfterModelCallEvent):
        if event.exception:
            event.retry = True

    async def another_retry_enabler(event: AfterModelCallEvent):
        if event.exception:
            event.retry = True

    agent = Agent(model=model)
    agent.hooks.add_callback(AfterModelCallEvent, retry_enabler)
    agent.hooks.add_callback(AfterModelCallEvent, another_retry_enabler)

    result = agent("Test multiple hooks")

    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Success"


@pytest.mark.asyncio
async def test_hook_retry_last_hook_wins(alist, mock_sleep):
    """Test that when multiple hooks set retry_model, the last-called hook wins.

    Note: AfterModelCallEvent callbacks are invoked in reverse order, so the first
    registered hook is called last.
    """

    class CustomException(Exception):
        pass

    call_count = [0]

    def mock_stream(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise CustomException("First attempt fails")
        else:
            raise CustomException(f"Should not be called (call {call_count[0]})")

    model = MagicMock()
    model.stream = mock_stream

    async def retry_enabler(event: AfterModelCallEvent):
        """Called first due to reverse order."""
        if event.exception:
            event.retry = True

    async def retry_disabler(event: AfterModelCallEvent):
        """Called last, so it wins."""
        if event.exception:
            event.retry = False

    agent = Agent(model=model)
    agent.hooks.add_callback(AfterModelCallEvent, retry_disabler)  # Registered first, called last
    agent.hooks.add_callback(AfterModelCallEvent, retry_enabler)  # Registered second, called first

    # Should raise exception since last-called hook disabled retry
    with pytest.raises(CustomException, match="First attempt fails"):
        agent("Test last hook wins")

    # Verify stream was only called once
    assert call_count[0] == 1


@pytest.mark.asyncio
async def test_hook_retry_with_throttle_exception(alist, mock_sleep):
    """Test that hook retry works alongside existing throttle retry."""

    class CustomException(Exception):
        pass

    model = MagicMock()
    model.stream.side_effect = [
        CustomException("Custom error"),
        ModelThrottledException("ThrottlingException"),
        ModelThrottledException("ThrottlingException"),
        MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [{"text": "Success after mixed retries"}],
                },
            ]
        ).stream([]),
    ]

    async def handle_after_model_call(event: AfterModelCallEvent):
        if event.exception and isinstance(event.exception, CustomException):
            event.retry = True

    agent = Agent(model=model)
    agent.hooks.add_callback(AfterModelCallEvent, handle_after_model_call)

    result = agent("Test mixed retries")

    # Should succeed after: custom retry + 2 throttle retries
    assert result.stop_reason == "end_turn"
    assert result.message["content"][0]["text"] == "Success after mixed retries"
