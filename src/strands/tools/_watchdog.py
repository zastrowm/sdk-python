import logging
from pathlib import Path
from typing import Any

from watchdog.events import FileSystemEventHandler

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


class ToolChangeHandler(FileSystemEventHandler):
    """Handler for tool file changes."""

    def __init__(self, tool_registry: ToolRegistry) -> None:
        """Initialize a tool change handler.

        Args:
            tool_registry: The tool registry to update when tools change.
        """
        self.tool_registry = tool_registry

    def on_modified(self, event: Any) -> None:
        """Reload tool if file modification detected.

        Args:
            event: The file system event that triggered this handler.
        """
        if event.src_path.endswith(".py"):
            tool_path = Path(event.src_path)
            tool_name = tool_path.stem

            if tool_name not in ["__init__"]:
                logger.debug("tool_name=<%s> | tool change detected", tool_name)
                try:
                    self.tool_registry.reload_tool(tool_name)
                except Exception as e:
                    logger.error("tool_name=<%s>, exception=<%s> | failed to reload tool", tool_name, str(e))


class MasterChangeHandler(FileSystemEventHandler):
    """Master handler that delegates to all registered handlers."""

    def __init__(self, dir_path: str) -> None:
        """Initialize a master change handler for a specific directory.

        Args:
            dir_path: The directory path to watch.
        """
        self.dir_path = dir_path

    def on_modified(self, event: Any) -> None:
        """Delegate file modification events to all registered handlers.

        Args:
            event: The file system event that triggered this handler.
        """
        from .watcher import ToolWatcher

        if event.src_path.endswith(".py"):
            tool_path = Path(event.src_path)
            tool_name = tool_path.stem

            if tool_name not in ["__init__"]:
                # Delegate to all registered handlers for this directory
                for handler in ToolWatcher._registry_handlers.get(self.dir_path, {}).values():
                    try:
                        handler.on_modified(event)
                    except Exception as e:
                        logger.error("exception=<%s> | handler error", str(e))
