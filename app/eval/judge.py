import json
from typing import List, Optional
from app.config import settings
from app.eval.schemas import DEFAULT_CRITERIA, EvaluationResult
from app.router.router import ModelRouter, model_router
from app.schemas.chat import ChatCompletionRequest, ChatMessage


class JudgeParsingError(ValueError):
    """Raised when the judge model's output isn't valid, well-formed JSON."""


def _build_judge_prompt(prompt: str, response: str, criteria: List[str]) -> str:
    criteria_list = ", ".join(criteria)
    return (
        "You are an impartial evaluator. Score the ASSISTANT RESPONSE to the USER PROMPT "
        f"on each of these dimensions, 1 (poor) to 5 (excellent): {criteria_list}.\n\n"
        f"USER PROMPT:\n{prompt}\n\nASSISTANT RESPONSE:\n{response}\n\n"
        "Reply with ONLY a JSON object, no other text, in this exact shape: "
        '{"scores": {"<dimension>": <1-5 int>, ...}, "rationale": "<one or two sentence explanation>"}'
    )


class Judge:
    """LLM-as-a-Judge: reference-free quality scoring of a (prompt, response) pair.

    Reuses the existing provider/router infrastructure to call the judge model --
    no separate LLM-calling mechanism. Standalone (POST /v1/evaluate), not run
    automatically on chat completions: judging is a second real LLM call, so it
    shouldn't silently double the cost/latency of every request.
    """

    def __init__(self, router: Optional[ModelRouter] = None, judge_model: Optional[str] = None):
        self._router = router or model_router
        self._judge_model = judge_model or settings.EVAL_JUDGE_MODEL

    async def evaluate(self, prompt: str, response: str, criteria: Optional[List[str]] = None) -> EvaluationResult:
        dimensions = criteria or DEFAULT_CRITERIA
        judge_request = ChatCompletionRequest(
            model=self._judge_model,
            messages=[ChatMessage(role="user", content=_build_judge_prompt(prompt, response, dimensions))],
            temperature=0.0,
        )
        judge_response = await self._router.execute_with_fallback(judge_request)
        raw_text = judge_response.choices[0].message.get_text()

        try:
            parsed = json.loads(raw_text)
            scores = {str(k): int(v) for k, v in parsed["scores"].items()}
            if not scores:
                raise ValueError("judge returned an empty scores object")
            rationale = str(parsed.get("rationale", ""))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise JudgeParsingError(f"Judge model returned unparseable output: {raw_text[:200]!r}") from exc

        overall_score = round(sum(scores.values()) / len(scores), 2)
        return EvaluationResult(scores=scores, overall_score=overall_score, rationale=rationale)
