from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def get_health() -> HealthResponse:
    """Health check endpoint to verify gateway operational status."""
    return HealthResponse(
        status="healthy",
        service=settings.APP_NAME,
        environment=settings.ENVIRONMENT,
        version="0.1.0",
    )
