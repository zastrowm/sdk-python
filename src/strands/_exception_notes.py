"""Exception note utilities for Python 3.10+ compatibility."""

import sys

# add_note was add_note in 3.11 - we hoist to a constant for facilitate testing
supports_add_note = sys.version_info >= (3, 11)


def add_exception_note(exception: Exception, note: str) -> None:
    """Add a note to an exception, compatible with Python 3.10+.

    Uses add_note() in Python 3.11+ or modifies the exception message in Python 3.10.
    """
    if supports_add_note:
        # we ignore the mypy error because the version-check for add_note is extracted into a constant up above and
        # mypy doesn't detect that
        exception.add_note(note)  # type: ignore
    else:
        # For Python 3.10, append note to the exception message
        if hasattr(exception, "args") and exception.args:
            exception.args = (f"{exception.args[0]}\n{note}",) + exception.args[1:]
        else:
            exception.args = (note,)
