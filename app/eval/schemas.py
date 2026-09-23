from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.config import settings

DEFAULT_CRITERIA = ["relevance", "coherence", "helpfulness"]


class EvaluationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=settings.MAX_TOTAL_CONTENT_CHARS)
    response: str = Field(..., min_length=1, max_length=settings.MAX_TOTAL_CONTENT_CHARS)
    criteria: Optional[List[str]] = Field(
        default=None, description="Dimensions to score, 1-5 each; defaults to relevance/coherence/helpfulness"
    )


class EvaluationResult(BaseModel):
    scores: Dict[str, int]
    overall_score: float
    rationale: str
