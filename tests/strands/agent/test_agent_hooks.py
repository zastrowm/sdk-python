import unittest.mock
from unittest.mock import call

import pytest
from pydantic import BaseModel

import strands
from strands import Agent
from strands.experimental.hooks import AgentInitializedEvent, EndRequestEvent, StartRequestEvent
from strands.types.content import Messages
from tests.fixtures.mock_hook_provider import MockHookProvider
from tests.fixtures.mocked_model_provider import MockedModelProvider


@pytest.fixture
def hook_provider():
    return MockHookProvider([AgentInitializedEvent, StartRequestEvent, EndRequestEvent])


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

    # for now, hooks are private
    agent._hooks.add_hook(hook_provider)

    return agent


# mock the User(name='Jane Doe', age=30)
class User(BaseModel):
    """A user of the system."""

    name: str
    age: int


@unittest.mock.patch("strands.experimental.hooks.registry.HookRegistry.invoke_callbacks")
def test_agent_hooks__init__(mock_invoke_callbacks):
    """Verify that the AgentInitializedEvent is emitted on Agent construction."""
    agent = Agent()

    # Verify AgentInitialized event was invoked
    mock_invoke_callbacks.assert_called_once()
    assert mock_invoke_callbacks.call_args == call(AgentInitializedEvent(agent=agent))


def test_agent_hooks__call__(agent, hook_provider, agent_tool, tool_use):
    """Verify that the correct hook events are emitted as part of __call__."""

    agent("test message")

    events = hook_provider.get_events()
    assert len(events) == 2

    assert events.popleft() == StartRequestEvent(agent=agent)
    assert events.popleft() == EndRequestEvent(agent=agent)


@pytest.mark.asyncio
async def test_agent_hooks_stream_async(agent, hook_provider, agent_tool, tool_use):
    """Verify that the correct hook events are emitted as part of stream_async."""
    iterator = agent.stream_async("test message")
    await anext(iterator)
    assert hook_provider.events_received == [StartRequestEvent(agent=agent)]

    # iterate the rest
    async for _ in iterator:
        pass

    events = hook_provider.get_events()
    assert len(events) == 2

    assert events.popleft() == StartRequestEvent(agent=agent)
    assert events.popleft() == EndRequestEvent(agent=agent)


def test_agent_hooks_structured_output(agent, hook_provider, agenerator):
    """Verify that the correct hook events are emitted as part of structured_output."""

    expected_user = User(name="Jane Doe", age=30)
    agent.model.structured_output = unittest.mock.Mock(return_value=agenerator([{"output": expected_user}]))
    agent.structured_output(User, "example prompt")

    assert hook_provider.events_received == [StartRequestEvent(agent=agent), EndRequestEvent(agent=agent)]
