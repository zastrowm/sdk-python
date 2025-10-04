"""SDK model providers.

This package includes an abstract base Model class along with concrete implementations for specific providers.
"""

import sys

from . import model
from .model import Model

# We don't build bedrock when targeting web-browsers
if sys.platform != "emscripten":
    from . import bedrock
    from .bedrock import BedrockModel

    __all__ = ["bedrock", "model", "BedrockModel", "Model"]
else:
    __all__ = ["model", "Model"]
