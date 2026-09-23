from fastapi import APIRouter, Depends, status
from app.core.rate_limit import enforce_rate_limit
from app.core.security import verify_api_key
from app.rag.document_store import document_store
from app.schemas.documents import DocumentIngestRequest, DocumentIngestResponse

router = APIRouter(prefix="/v1", tags=["Documents"], dependencies=[Depends(enforce_rate_limit), Depends(verify_api_key)])


@router.post("/documents", response_model=DocumentIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(payload: DocumentIngestRequest) -> DocumentIngestResponse:
    """Chunk, embed, and store a document for later retrieval-augmented chat requests."""
    document_id, chunks_created = await document_store.ingest_document(payload.text)
    return DocumentIngestResponse(document_id=document_id, chunks_created=chunks_created)
