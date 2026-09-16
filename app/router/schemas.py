from enum import Enum
from typing import Optional
from pydantic import BaseModel


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class RequestType(str, Enum):
    TEXT = "text"
    VISION = "vision"


class RoutingDecision(BaseModel):
    selected_model: str
    selected_provider_name: str
    complexity: ComplexityLevel
    request_type: RequestType
    reason: str
    fallback_model: Optional[str] = None
    estimated_cost_usd: float = 0.0
