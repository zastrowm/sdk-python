"""Sliding window conversation history management."""

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ...agent.agent import Agent

from ...types.content import Message, Messages
from ...types.exceptions import ContextWindowOverflowException
from .conversation_manager import ConversationManager

logger = logging.getLogger(__name__)


def is_user_message(message: Message) -> bool:
    """Check if a message is from a user.

    Args:
        message: The message object to check.

    Returns:
        True if the message has the user role, False otherwise.
    """
    return message["role"] == "user"


def is_assistant_message(message: Message) -> bool:
    """Check if a message is from an assistant.

    Args:
        message: The message object to check.

    Returns:
        True if the message has the assistant role, False otherwise.
    """
    return message["role"] == "assistant"


class SlidingWindowConversationManager(ConversationManager):
    """Implements a sliding window strategy for managing conversation history.

    This class handles the logic of maintaining a conversation window that preserves tool usage pairs and avoids
    invalid window states.
    """

    def __init__(self, window_size: int = 40, should_truncate_results: bool = True):
        """Initialize the sliding window conversation manager.

        Args:
            window_size: Maximum number of messages to keep in the agent's history.
                Defaults to 40 messages.
            should_truncate_results: Truncate tool results when a message is too large for the model's context window
        """
        self.window_size = window_size
        self.should_truncate_results = should_truncate_results

    def apply_management(self, agent: "Agent") -> None:
        """Apply the sliding window to the agent's messages array to maintain a manageable history size.

        This method is called after every event loop cycle, as the messages array may have been modified with tool
        results and assistant responses. It first removes any dangling messages that might create an invalid
        conversation state, then applies the sliding window if the message count exceeds the window size.

        Special handling is implemented to ensure we don't leave a user message with toolResult
        as the first message in the array. It also ensures that all toolUse blocks have corresponding toolResult
        blocks to maintain conversation coherence.

        Args:
            agent: The agent whose messages will be managed.
                This list is modified in-place.
        """
        messages = agent.messages
        self._remove_dangling_messages(messages)

        if len(messages) <= self.window_size:
            logger.debug(
                "window_size=<%s>, message_count=<%s> | skipping context reduction", len(messages), self.window_size
            )
            return
        self.reduce_context(agent)

    def _remove_dangling_messages(self, messages: Messages) -> None:
        """Remove dangling messages that would create an invalid conversation state.

        After the event loop cycle is executed, we expect the messages array to end with either an assistant tool use
        request followed by the pairing user tool result or an assistant response with no tool use request. If the
        event loop cycle fails, we may end up in an invalid message state, and so this method will remove problematic
        messages from the end of the array.

        This method handles two specific cases:

        - User with no tool result: Indicates that event loop failed to generate an assistant tool use request
        - Assistant with tool use request: Indicates that event loop failed to generate a pairing user tool result

        Args:
            messages: The messages to clean up.
                This list is modified in-place.
        """
        # remove any dangling user messages with no ToolResult
        if len(messages) > 0 and is_user_message(messages[-1]):
            if not any("toolResult" in content for content in messages[-1]["content"]):
                messages.pop()

        # remove any dangling assistant messages with ToolUse
        if len(messages) > 0 and is_assistant_message(messages[-1]):
            if any("toolUse" in content for content in messages[-1]["content"]):
                messages.pop()
                # remove remaining dangling user messages with no ToolResult after we popped off an assistant message
                if len(messages) > 0 and is_user_message(messages[-1]):
                    if not any("toolResult" in content for content in messages[-1]["content"]):
                        messages.pop()

    def reduce_context(self, agent: "Agent", e: Optional[Exception] = None) -> None:
        """Trim the oldest messages to reduce the conversation context size.

        The method handles special cases where trimming the messages leads to:
         - toolResult with no corresponding toolUse
         - toolUse with no corresponding toolResult

        Args:
            agent: The agent whose messages will be reduce.
                This list is modified in-place.
            e: The exception that triggered the context reduction, if any.

        Raises:
            ContextWindowOverflowException: If the context cannot be reduced further.
                Such as when the conversation is already minimal or when tool result messages cannot be properly
                converted.
        """
        messages = agent.messages

        # Try to truncate the tool result first
        last_message_idx_with_tool_results = self._find_last_message_with_tool_results(messages)
        if last_message_idx_with_tool_results is not None and self.should_truncate_results:
            logger.debug(
                "message_index=<%s> | found message with tool results at index", last_message_idx_with_tool_results
            )
            results_truncated = self._truncate_tool_results(messages, last_message_idx_with_tool_results)
            if results_truncated:
                logger.debug("message_index=<%s> | tool results truncated", last_message_idx_with_tool_results)
                return

        # Try to trim index id when tool result cannot be truncated anymore
        # If the number of messages is less than the window_size, then we default to 2, otherwise, trim to window size
        trim_index = 2 if len(messages) <= self.window_size else len(messages) - self.window_size

        # Find the next valid trim_index
        while trim_index < len(messages):
            if (
                # Oldest message cannot be a toolResult because it needs a toolUse preceding it
                any("toolResult" in content for content in messages[trim_index]["content"])
                or (
                    # Oldest message can be a toolUse only if a toolResult immediately follows it.
                    any("toolUse" in content for content in messages[trim_index]["content"])
                    and trim_index + 1 < len(messages)
                    and not any("toolResult" in content for content in messages[trim_index + 1]["content"])
                )
            ):
                trim_index += 1
            else:
                break
        else:
            # If we didn't find a valid trim_index, then we throw
            raise ContextWindowOverflowException("Unable to trim conversation context!") from e

        # Overwrite message history
        messages[:] = messages[trim_index:]

    def _truncate_tool_results(self, messages: Messages, msg_idx: int) -> bool:
        """Truncate tool results in a message to reduce context size.

        When a message contains tool results that are too large for the model's context window, this function
        replaces the content of those tool results with a simple error message.

        Args:
            messages: The conversation message history.
            msg_idx: Index of the message containing tool results to truncate.

        Returns:
            True if any changes were made to the message, False otherwise.
        """
        if msg_idx >= len(messages) or msg_idx < 0:
            return False

        message = messages[msg_idx]
        changes_made = False
        tool_result_too_large_message = "The tool result was too large!"
        for i, content in enumerate(message.get("content", [])):
            if isinstance(content, dict) and "toolResult" in content:
                tool_result_content_text = next(
                    (item["text"] for item in content["toolResult"]["content"] if "text" in item),
                    "",
                )
                # make the overwriting logic togglable
                if (
                    message["content"][i]["toolResult"]["status"] == "error"
                    and tool_result_content_text == tool_result_too_large_message
                ):
                    logger.info("ToolResult has already been updated, skipping overwrite")
                    return False
                # Update status to error with informative message
                message["content"][i]["toolResult"]["status"] = "error"
                message["content"][i]["toolResult"]["content"] = [{"text": tool_result_too_large_message}]
                changes_made = True

        return changes_made

    def _find_last_message_with_tool_results(self, messages: Messages) -> Optional[int]:
        """Find the index of the last message containing tool results.

        This is useful for identifying messages that might need to be truncated to reduce context size.

        Args:
            messages: The conversation message history.

        Returns:
            Index of the last message with tool results, or None if no such message exists.
        """
        # Iterate backwards through all messages (from newest to oldest)
        for idx in range(len(messages) - 1, -1, -1):
            # Check if this message has any content with toolResult
            current_message = messages[idx]
            has_tool_result = False

            for content in current_message.get("content", []):
                if isinstance(content, dict) and "toolResult" in content:
                    has_tool_result = True
                    break

            if has_tool_result:
                return idx

        return None
