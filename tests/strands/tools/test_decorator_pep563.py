"""Tests for PEP 563 (from __future__ import annotations) compatibility.

This module tests that the @tool decorator works correctly when modules use
`from __future__ import annotations` (PEP 563), which causes all annotations
to be stored as string literals rather than evaluated types.

This is a regression test for issue #1208:
https://github.com/strands-agents/sdk-python/issues/1208
"""

from __future__ import annotations

from typing import Any

import pytest
from typing_extensions import Literal, TypedDict

from strands import tool

# Define types at module level (simulating nova-act pattern)
CLICK_TYPE = Literal["left", "right", "middle", "double"]
EXTRA_TYPE = Literal["extra"]


class ClickOptions(TypedDict):
    """Options for click operation."""

    blur_field: bool | None


@tool
def simple_literal_tool(click_type: CLICK_TYPE) -> dict[str, Any]:
    """A tool with a simple Literal type parameter.

    Args:
        click_type: The type of click to perform.

    Returns:
        Result dictionary.
    """
    return {"status": "success", "content": [{"text": f"Clicked: {click_type}"}]}


@tool
def complex_literal_tool(
    box: str,
    extra: EXTRA_TYPE,
    click_type: CLICK_TYPE | None = None,
    click_options: ClickOptions | None = None,
) -> Any:
    """A tool with complex type hints including Literal and TypedDict.

    This simulates the nova-act browser click tool pattern.

    Args:
        box: The box to click.
        extra: Extra type parameter.
        click_type: The type of click to perform.
        click_options: Additional click options.

    Returns:
        Result string.
    """
    return "Done"


@tool
def union_literal_tool(mode: Literal["fast", "slow"] | None = None) -> str:
    """A tool with inline Literal in Union.

    Args:
        mode: The execution mode.

    Returns:
        Status string.
    """
    return f"Mode: {mode}"


class TestPEP563Compatibility:
    """Test suite for PEP 563 (future annotations) compatibility."""

    def test_simple_literal_type_tool_spec(self):
        """Test that simple Literal type parameters work with __future__ annotations."""
        spec = simple_literal_tool.tool_spec
        assert spec["name"] == "simple_literal_tool"

        schema = spec["inputSchema"]["json"]
        assert "click_type" in schema["properties"]
        # Verify Literal values are present in schema
        click_type_schema = schema["properties"]["click_type"]
        assert "enum" in click_type_schema or "anyOf" in click_type_schema

    def test_complex_literal_type_tool_spec(self):
        """Test that complex type hints with Literal work with __future__ annotations."""
        spec = complex_literal_tool.tool_spec
        assert spec["name"] == "complex_literal_tool"

        schema = spec["inputSchema"]["json"]
        # All parameters should be present
        assert "box" in schema["properties"]
        assert "extra" in schema["properties"]
        assert "click_type" in schema["properties"]
        assert "click_options" in schema["properties"]

        # Required parameters should be marked
        assert "box" in schema["required"]
        assert "extra" in schema["required"]

    def test_union_literal_tool_spec(self):
        """Test that inline Literal in Union works with __future__ annotations."""
        spec = union_literal_tool.tool_spec
        assert spec["name"] == "union_literal_tool"

        schema = spec["inputSchema"]["json"]
        assert "mode" in schema["properties"]

    def test_simple_literal_tool_invocation(self):
        """Test that tools with Literal types can be invoked."""
        result = simple_literal_tool(click_type="left")
        assert result["status"] == "success"
        assert "left" in result["content"][0]["text"]

    def test_complex_literal_tool_invocation(self):
        """Test that tools with complex types can be invoked."""
        result = complex_literal_tool(
            box="box1",
            extra="extra",
            click_type="double",
            click_options={"blur_field": True},
        )
        assert result == "Done"

    def test_tool_spec_no_pydantic_error(self):
        """Verify no PydanticUserError is raised when accessing tool_spec.

        This is the specific error from issue #1208:
        PydanticUserError: `Agent_clickTool` is not fully defined;
        you should define `EXTRA_TYPE`, then call `Agent_clickTool.model_rebuild()`.
        """
        # This should not raise PydanticUserError
        try:
            _ = simple_literal_tool.tool_spec
            _ = complex_literal_tool.tool_spec
            _ = union_literal_tool.tool_spec
        except Exception as e:
            if "not fully defined" in str(e):
                pytest.fail(f"PydanticUserError raised - PEP 563 compatibility broken: {e}")
            raise
