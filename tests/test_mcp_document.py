import pytest
import app.mcp.document_server as document_server
from app.rag.document_store import DocumentStore
from tests.test_semantic_cache import FakeEmbeddingClient


@pytest.fixture
def fake_store(monkeypatch):
    fake_client = FakeEmbeddingClient(
        vectors_by_text={
            "Our office hours are 9am to 5pm.": [1.0, 0.0],
            "When are you open?": [0.99, 0.01],
        }
    )
    store = DocumentStore(embedding_client=fake_client)
    monkeypatch.setattr(document_server, "document_store", store)
    return store


@pytest.mark.asyncio
async def test_ingest_document_tool_stores_and_returns_metadata(fake_store):
    result = await document_server.ingest_document("Our office hours are 9am to 5pm.")
    assert result["chunks_created"] == 1
    assert result["document_id"]


@pytest.mark.asyncio
async def test_search_documents_tool_returns_relevant_chunks(fake_store):
    await document_server.ingest_document("Our office hours are 9am to 5pm.")
    results = await document_server.search_documents("When are you open?", top_k=1)
    assert results == ["Our office hours are 9am to 5pm."]


@pytest.mark.asyncio
async def test_search_documents_tool_empty_store_returns_nothing(fake_store):
    results = await document_server.search_documents("anything")
    assert results == []
