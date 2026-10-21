class LLMError(Exception):
    """Base exception for LLM client failures."""


class LLMUnavailableError(LLMError):
    def __init__(self) -> None:
        super().__init__("Model service is temporarily unavailable")


class LLMInvalidResponseError(LLMError):
    def __init__(self) -> None:
        super().__init__("Model service returned an invalid response")
