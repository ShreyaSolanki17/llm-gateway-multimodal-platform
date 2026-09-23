from fastapi import APIRouter, Depends, status
from app.core.exceptions import EvaluationError
from app.core.rate_limit import enforce_rate_limit
from app.core.security import verify_api_key
from app.eval.judge import Judge, JudgeParsingError
from app.eval.schemas import EvaluationRequest, EvaluationResult

router = APIRouter(prefix="/v1", tags=["Evaluation"], dependencies=[Depends(enforce_rate_limit), Depends(verify_api_key)])
_judge = Judge()


@router.post("/evaluate", response_model=EvaluationResult, status_code=status.HTTP_200_OK)
async def evaluate_response(payload: EvaluationRequest) -> EvaluationResult:
    """Score a (prompt, response) pair via LLM-as-a-Judge, reference-free quality scoring."""
    try:
        return await _judge.evaluate(payload.prompt, payload.response, payload.criteria)
    except JudgeParsingError as exc:
        raise EvaluationError(str(exc))
