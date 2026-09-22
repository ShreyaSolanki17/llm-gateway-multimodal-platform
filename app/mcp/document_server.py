from typing import Any, Dict, List
from mcp.server.mcpserver import MCPServer
from app.rag.document_store import document_store

server = MCPServer(
    name="document-mcp-server",
    instructions="Search and add to the gateway's shared document knowledge base (the same store the chat endpoint's use_rag flag reads from).",
)


@server.tool()
async def search_documents(query: str, top_k: int = 3) -> List[str]:
    """Return the most relevant document chunks for a query."""
    return await document_store.retrieve(query, top_k=top_k)


@server.tool()
async def ingest_document(text: str) -> Dict[str, Any]:
    """Chunk, embed, and store a document for future retrieval."""
    document_id, chunks_created = await document_store.ingest_document(text)
    return {"document_id": document_id, "chunks_created": chunks_created}


if __name__ == "__main__":
    server.run()
