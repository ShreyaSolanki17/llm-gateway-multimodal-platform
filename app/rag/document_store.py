import uuid
from dataclasses import dataclass
from typing import List, Optional
from app.cache.embeddings import EmbeddingClient
from app.config import settings
from app.core.logging import logger
from app.core.similarity import cosine_similarity
from app.rag.chunker import chunk_text


@dataclass
class Chunk:
    document_id: str
    text: str
    embedding: List[float]


class DocumentStore:
    """In-memory RAG knowledge base: chunks documents, embeds each chunk, and
    retrieves the top-k most similar chunks for a query.

    ponytail: in-process list, O(n) scan per retrieval. Fine at prototype scale;
    swap for pgvector (Milestone 17) once the corpus outgrows a linear scan.
    """

    def __init__(self, embedding_client: Optional[EmbeddingClient] = None):
        self._embedding_client = embedding_client or EmbeddingClient()
        self._chunks: List[Chunk] = []

    async def ingest_document(self, text: str) -> tuple[str, int]:
        """Chunk, embed, and store a document. Returns (document_id, chunks_created)."""
        document_id = uuid.uuid4().hex
        pieces = chunk_text(text, settings.RAG_CHUNK_SIZE, settings.RAG_CHUNK_OVERLAP)

        stored = 0
        for piece in pieces:
            try:
                embedding = await self._embedding_client.embed(piece)
            except Exception as exc:
                logger.warning(f"RAG ingest skipped a chunk, embedding failed: {str(exc)}")
                continue
            self._chunks.append(Chunk(document_id=document_id, text=piece, embedding=embedding))
            stored += 1

        logger.info(f"Ingested document '{document_id}' with {stored}/{len(pieces)} chunks stored")
        return document_id, stored

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Return the top-k most similar chunk texts for a query, best match first."""
        if not query or not self._chunks:
            return []

        try:
            query_embedding = await self._embedding_client.embed(query)
        except Exception as exc:
            logger.warning(f"RAG retrieval skipped, embedding failed: {str(exc)}")
            return []

        k = top_k if top_k is not None else settings.RAG_TOP_K
        scored = sorted(
            self._chunks,
            key=lambda c: cosine_similarity(query_embedding, c.embedding),
            reverse=True,
        )
        return [c.text for c in scored[:k]]


# Global Singleton Document Store Instance
document_store = DocumentStore()
