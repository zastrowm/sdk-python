from typing import Callable

from ..models import Model


def create_bedrock_model(model_id: str | None) -> Model:
    from ..models.bedrock import BedrockModel

    return (
        BedrockModel() if not model_id else BedrockModel(model_id=model_id) if isinstance(model_id, str) else model_id
    )


def create_printable_callback_handler() -> Callable:
    from ..handlers import PrintingCallbackHandler

    return PrintingCallbackHandler()


class AgentPropertyFactories:
    """Factory methods for constructing properties."""

    create_model: Callable[[str | None], Model] = create_bedrock_model

    create_callback_handler: Callable[[], Callable] = create_printable_callback_handler
