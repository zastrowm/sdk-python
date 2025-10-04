"""Tool watcher for hot reloading tools during development.

This module provides functionality to watch tool directories for changes and automatically reload tools when they are
modified.
"""

import logging
from typing import TYPE_CHECKING, Dict, Set

from .registry import ToolRegistry

if TYPE_CHECKING:
    from ._watchdog import ToolChangeHandler

logger = logging.getLogger(__name__)


class ToolWatcher:
    """Watches tool directories for changes and reloads tools when they are modified."""

    # This class uses class variables for the observer and handlers because watchdog allows only one Observer instance
    # per directory. Using class variables ensures that all ToolWatcher instances share a single Observer, with the
    # MasterChangeHandler routing file system events to the appropriate individual handlers for each registry. This
    # design pattern avoids conflicts when multiple tool registries are watching the same directories.

    _shared_observer = None
    _watched_dirs: Set[str] = set()
    _observer_started = False
    _registry_handlers: Dict[str, Dict[int, "ToolChangeHandler"]] = {}

    def __init__(self, tool_registry: ToolRegistry) -> None:
        """Initialize a tool watcher for the given tool registry.

        Args:
            tool_registry: The tool registry to report changes.
        """
        self.tool_registry = tool_registry
        self.start()

    def start(self) -> None:
        """Start watching all tools directories for changes."""
        from watchdog.observers import Observer

        from ._watchdog import MasterChangeHandler, Observer, ToolChangeHandler

        # Initialize shared observer if not already done
        if ToolWatcher._shared_observer is None:
            ToolWatcher._shared_observer = Observer()

        # Create handler for this instance
        self.tool_change_handler = ToolChangeHandler(self.tool_registry)
        registry_id = id(self.tool_registry)

        # Get tools directories to watch
        tools_dirs = self.tool_registry.get_tools_dirs()

        for tools_dir in tools_dirs:
            dir_str = str(tools_dir)

            # Initialize the registry handlers dict for this directory if needed
            if dir_str not in ToolWatcher._registry_handlers:
                ToolWatcher._registry_handlers[dir_str] = {}

            # Store this handler with its registry id
            ToolWatcher._registry_handlers[dir_str][registry_id] = self.tool_change_handler

            # Schedule or update the master handler for this directory
            if dir_str not in ToolWatcher._watched_dirs:
                # First time seeing this directory, create a master handler
                master_handler = MasterChangeHandler(dir_str)
                ToolWatcher._shared_observer.schedule(master_handler, dir_str, recursive=False)
                ToolWatcher._watched_dirs.add(dir_str)
                logger.debug("tools_dir=<%s> | started watching tools directory", tools_dir)
            else:
                # Directory already being watched, just log it
                logger.debug("tools_dir=<%s> | directory already being watched", tools_dir)

        # Start the observer if not already started
        if not ToolWatcher._observer_started:
            ToolWatcher._shared_observer.start()
            ToolWatcher._observer_started = True
            logger.debug("tool directory watching initialized")
