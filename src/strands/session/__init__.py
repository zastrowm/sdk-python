"""Session module.

This module provides session management functionality.
"""

import sys

from .session_manager import SessionManager
from .session_repository import SessionRepository

if sys.platform != "emscripten":
    from .file_session_manager import FileSessionManager
    from .repository_session_manager import RepositorySessionManager
    from .s3_session_manager import S3SessionManager

    __all__ = [
        "FileSessionManager",
        "RepositorySessionManager",
        "S3SessionManager",
        "SessionManager",
        "SessionRepository",
    ]
else:
    __all__ = [
        "SessionManager",
        "SessionRepository",
    ]
