from pydantic import BaseModel, Field
from app.config import settings


class DocumentIngestRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=settings.MAX_DOCUMENT_CHARS, description="Raw document text to chunk, embed, and store"
    )


class DocumentIngestResponse(BaseModel):
    document_id: str
    chunks_created: int
