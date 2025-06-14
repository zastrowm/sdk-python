
import logging
from concurrent.futures import TimeoutError
from typing import Callable, List, Optional, Tuple, Protocol, Generator, TYPE_CHECKING

from ..types.tools import ToolResult, ToolUse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..types.event_loop import ParallelToolExecutorInterface

class ToolExecutorStrategy(Protocol):

    def run_tools(
            self,
            tool_uses: List[ToolUse],
            handler: Callable[[ToolUse], Tuple[bool, Optional[ToolResult]]]
    ) -> Generator[Tuple[bool, Optional[ToolResult]], None, None]:
        ...

class SequentialToolExecutorStrategy(ToolExecutorStrategy):
    def run_tools(
            self,
            tool_uses: List[ToolUse],
            handler: Callable[[ToolUse], Tuple[bool, Optional[ToolResult]]]
    ) -> Generator[Tuple[bool, Optional[ToolResult]], None, None]:
        """Execute tools sequentially and yield results.

        Args:
            tool_uses: List of tool uses to execute
            handler: Callback function to handle each tool execution

        Yields:
            Tuple containing:
                - bool indicating if tool succeeded
                - Optional tool result
        """
        for tool_use in tool_uses:
            succeeded, result = handler(tool_use)
            yield succeeded, result


class ParallelToolExecutorStrategy(ToolExecutorStrategy):
    def __init__(self, parallel_executor: "ParallelToolExecutorInterface"):
        """Initialize with a parallel executor.

        Args:
            parallel_executor: The executor to use for parallel tool execution
        """
        self.parallel_executor = parallel_executor

    def run_tools(
        self,
        tool_uses: List[ToolUse],
        handler: Callable[[ToolUse], Tuple[bool, Optional[ToolResult]]]
    ) -> Generator[Tuple[bool, Optional[ToolResult]], None, None]:
        """Execute tools in parallel and yield results.

        Args:
            tool_uses: List of tool uses to execute
            handler: Callback function to handle each tool execution

        Yields:
            Tuple containing:
                - bool indicating if tool succeeded
                - Optional tool result
        """
        logger.debug(
            "tool_count=<%s>, tool_executor=<%s> | executing tools in parallel",
            len(tool_uses),
            type(self.parallel_executor).__name__,
        )

        # Submit all tasks with their associated tools
        future_to_tool = {
            self.parallel_executor.submit(handler, tool_use): tool_use for tool_use in tool_uses
        }
        logger.debug("tool_count=<%s> | submitted tasks to parallel executor", len(tool_uses))

        # Collect results in parallel using the provided executor's as_completed method
        try:
            for future in self.parallel_executor.as_completed(future_to_tool):
                try:
                    succeeded, result = future.result()
                    yield succeeded, result
                except Exception as e:
                    tool = future_to_tool[future]
                    logger.debug("tool_name=<%s> | tool execution failed | %s", tool["name"], e)
                    yield False, None
        except TimeoutError:
            logger.error("timeout_seconds=<%s> | parallel tool execution timed out", self.parallel_executor.timeout)
            # Process any completed tasks
            for future in future_to_tool:
                if future.done():  # type: ignore
                    try:
                        succeeded, result = future.result(timeout=0)
                        yield succeeded, result
                    except Exception as tool_e:
                        tool = future_to_tool[future]
                        logger.debug("tool_name=<%s> | tool execution failed | %s", tool["name"], tool_e)
                        yield False, None
                else:
                    # This future didn't complete within the timeout
                    tool = future_to_tool[future]
                    logger.debug("tool_name=<%s> | tool execution timed out", tool["name"])
                    yield False, None