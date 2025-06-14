"""This module provides utilities for processing and manipulating conversation messages within the event loop.

It includes functions for cleaning up orphaned tool uses, finding messages with specific content types, and truncating
large tool results to prevent context window overflow.
"""

import logging
from typing import Dict, Set, Tuple

from ..types.content import Messages

logger = logging.getLogger(__name__)


def clean_orphaned_empty_tool_uses(messages: Messages) -> bool:
    """Clean up orphaned empty tool uses in conversation messages.

    This function identifies and removes any toolUse entries with empty input that don't have a corresponding
    toolResult. This prevents validation errors that occur when the model expects matching toolResult blocks for each
    toolUse.

    The function applies fixes by either:

    1. Replacing a message containing only an orphaned toolUse with a context message
    2. Removing the orphaned toolUse entry from a message with multiple content items

    Args:
        messages: The conversation message history.

    Returns:
        True if any fixes were applied, False otherwise.
    """
    if not messages:
        return False

    # Dictionary to track empty toolUse entries: {tool_id: (msg_index, content_index, tool_name)}
    empty_tool_uses: Dict[str, Tuple[int, int, str]] = {}

    # Set to track toolResults that have been seen
    tool_results: Set[str] = set()

    # Identify empty toolUse entries
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue

        for j, content in enumerate(msg.get("content", [])):
            if isinstance(content, dict) and "toolUse" in content:
                tool_use = content.get("toolUse", {})
                tool_id = tool_use.get("toolUseId")
                tool_input = tool_use.get("input", {})
                tool_name = tool_use.get("name", "unknown tool")

                # Check if this is an empty toolUse
                if tool_id and (not tool_input or tool_input == {}):
                    empty_tool_uses[tool_id] = (i, j, tool_name)

    # Identify toolResults
    for msg in messages:
        if msg.get("role") != "user":
            continue

        for content in msg.get("content", []):
            if isinstance(content, dict) and "toolResult" in content:
                tool_result = content.get("toolResult", {})
                tool_id = tool_result.get("toolUseId")
                if tool_id:
                    tool_results.add(tool_id)

    # Filter for orphaned empty toolUses (no corresponding toolResult)
    orphaned_tool_uses = {tool_id: info for tool_id, info in empty_tool_uses.items() if tool_id not in tool_results}

    # Apply fixes in reverse order of occurrence (to avoid index shifting)
    if not orphaned_tool_uses:
        return False

    # Sort by message index and content index in reverse order
    sorted_orphaned = sorted(orphaned_tool_uses.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

    # Apply fixes
    for tool_id, (msg_idx, content_idx, tool_name) in sorted_orphaned:
        logger.debug(
            "tool_name=<%s>, tool_id=<%s>, message_index=<%s>, content_index=<%s> "
            "fixing orphaned empty tool use at message index",
            tool_name,
            tool_id,
            msg_idx,
            content_idx,
        )
        try:
            # Check if this is the sole content in the message
            if len(messages[msg_idx]["content"]) == 1:
                # Replace with a message indicating the attempted tool
                messages[msg_idx]["content"] = [{"text": f"[Attempted to use {tool_name}, but operation was canceled]"}]
                logger.debug("message_index=<%s> | replaced content with context message", msg_idx)
            else:
                # Simply remove the orphaned toolUse entry
                messages[msg_idx]["content"].pop(content_idx)
                logger.debug(
                    "message_index=<%s>, content_index=<%s> | removed content item from message", msg_idx, content_idx
                )
        except Exception as e:
            logger.warning("failed to fix orphaned tool use | %s", e)

    return True
