class ServiceError(Exception):
    """Base exception for service-layer failures."""


class ConversationNotFoundError(ServiceError):
    def __init__(self, conversation_no: str) -> None:
        super().__init__(f"Conversation {conversation_no} was not found")
        self.conversation_no = conversation_no


class ConversationClosedError(ServiceError):
    def __init__(self, conversation_no: str) -> None:
        super().__init__(f"Conversation {conversation_no} is already closed")
        self.conversation_no = conversation_no
