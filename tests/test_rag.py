import pytest
from fastapi.testclient import TestClient
from app.rag.chunker import chunk_text
from app.rag.document_store import DocumentStore
from app.rag.augment import augment_with_context
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from tests.test_semantic_cache import FakeEmbeddingClient


def test_chunk_text_basic_split():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=4, overlap=0)
    assert chunks == ["word0 word1 word2 word3", "word4 word5 word6 word7", "word8 word9"]


def test_chunk_text_with_overlap():
    text = " ".join(f"word{i}" for i in range(10))
    chunks = chunk_text(text, chunk_size=4, overlap=2)
    assert chunks[0] == "word0 word1 word2 word3"
    assert chunks[1] == "word2 word3 word4 word5"


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("", chunk_size=200, overlap=50) == []


@pytest.mark.asyncio
async def test_document_store_ingest_and_retrieve():
    fake_client = FakeEmbeddingClient(
        vectors_by_text={
            "Refunds are processed within 5 business days.": [1.0, 0.0],
            "What is the refund policy?": [0.98, 0.02],
        }
    )
    store = DocumentStore(embedding_client=fake_client)

    document_id, chunks_created = await store.ingest_document("Refunds are processed within 5 business days.")
    assert chunks_created == 1
    assert document_id

    results = await store.retrieve("What is the refund policy?", top_k=1)
    assert results == ["Refunds are processed within 5 business days."]


@pytest.mark.asyncio
async def test_document_store_retrieve_empty_store_returns_nothing():
    store = DocumentStore(embedding_client=FakeEmbeddingClient())
    assert await store.retrieve("anything") == []


@pytest.mark.asyncio
async def test_document_store_embedding_failure_degrades_gracefully():
    store = DocumentStore(embedding_client=FakeEmbeddingClient(should_fail=True))
    document_id, chunks_created = await store.ingest_document("Some content that will fail to embed.")
    assert chunks_created == 0
    assert await store.retrieve("query") == []


@pytest.mark.asyncio
async def test_augment_with_context_prepends_system_message():
    fake_client = FakeEmbeddingClient(
        vectors_by_text={
            "Our office hours are 9am to 5pm.": [1.0, 0.0],
            "When are you open?": [0.99, 0.01],
        }
    )
    store = DocumentStore(embedding_client=fake_client)
    await store.ingest_document("Our office hours are 9am to 5pm.")

    request = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="When are you open?")])
    augmented = await augment_with_context(request, store=store)

    assert len(augmented.messages) == 2
    assert augmented.messages[0].role == "system"
    assert "office hours" in augmented.messages[0].get_text()
    assert augmented.messages[1] == request.messages[0]


@pytest.mark.asyncio
async def test_augment_with_context_returns_unchanged_when_no_match():
    store = DocumentStore(embedding_client=FakeEmbeddingClient())
    request = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="Anything")])
    augmented = await augment_with_context(request, store=store)
    assert augmented is request


def test_document_ingest_endpoint(client: TestClient, monkeypatch):
    import app.api.v1.documents as documents_module

    fake_store = DocumentStore(embedding_client=FakeEmbeddingClient())
    monkeypatch.setattr(documents_module, "document_store", fake_store)

    response = client.post("/v1/documents", json={"text": "Some policy text to ingest for retrieval testing."})
    assert response.status_code == 201
    data = response.json()
    assert data["document_id"]
    assert data["chunks_created"] == 1


def test_document_ingest_endpoint_rejects_empty_text(client: TestClient):
    response = client.post("/v1/documents", json={"text": ""})
    assert response.status_code == 422


def test_chat_endpoint_with_use_rag_augments_prompt(client: TestClient, monkeypatch):
    import asyncio

    fake_client = FakeEmbeddingClient(
        vectors_by_text={
            "Our refund window is 30 days.": [1.0, 0.0],
            "How long is the refund window?": [0.99, 0.01],
        }
    )
    fake_store = DocumentStore(embedding_client=fake_client)
    monkeypatch.setattr("app.rag.augment.document_store", fake_store)

    asyncio.run(fake_store.ingest_document("Our refund window is 30 days."))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-gpt-4o",
            "messages": [{"role": "user", "content": "How long is the refund window?"}],
            "use_rag": True,
        },
    )
    assert response.status_code == 200
