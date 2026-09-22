from typing import Optional
from app.rag.document_store import DocumentStore, document_store
from app.schemas.chat import ChatCompletionRequest, ChatMessage


async def augment_with_context(
    request: ChatCompletionRequest, store: Optional[DocumentStore] = None
) -> ChatCompletionRequest:
    """Retrieve relevant document chunks and prepend them as a system message.
    Returns the original request unchanged if nothing relevant is found."""
    store = store or document_store
    query = next((m.get_text() for m in reversed(request.messages) if m.role == "user"), "")
    chunks = await store.retrieve(query)
    if not chunks:
        return request

    context_message = ChatMessage(
        role="system",
        content="Use the following retrieved context to answer the user's question if relevant:\n\n"
        + "\n\n".join(chunks),
    )
    return request.model_copy(update={"messages": [context_message] + request.messages})
