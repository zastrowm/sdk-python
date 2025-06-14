import pytest

import strands
from strands import Agent
from strands.models.litellm import LiteLLMModel


@pytest.fixture
def model():
    return LiteLLMModel(model_id="bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0")


@pytest.fixture
def tools():
    @strands.tool
    def tool_time() -> str:
        return "12:00"

    @strands.tool
    def tool_weather() -> str:
        return "sunny"

    return [tool_time, tool_weather]


@pytest.fixture
def agent(model, tools):
    return Agent(model=model, tools=tools)


def test_agent(agent):
    result = agent("What is the time and weather in New York?")
    text = result.message["content"][0]["text"].lower()

    assert all(string in text for string in ["12:00", "sunny"])
