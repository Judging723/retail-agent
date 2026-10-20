class RepositoryError(Exception):
    """Base exception for repository failures."""


class ConversationNotFoundError(RepositoryError):
    def __init__(self, conversation_id: int) -> None:
        super().__init__(f"Conversation with id {conversation_id} was not found")
        self.conversation_id = conversation_id
