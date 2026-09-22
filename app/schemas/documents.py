from pydantic import BaseModel, Field


class DocumentIngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to chunk, embed, and store")


class DocumentIngestResponse(BaseModel):
    document_id: str
    chunks_created: int
