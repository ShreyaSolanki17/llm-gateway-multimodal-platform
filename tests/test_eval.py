import pytest
from fastapi.testclient import TestClient
from app.providers.base import BaseLLMProvider, ModelMetadata
from app.providers.registry import ProviderRegistry
from app.router.router import ModelRouter
from app.eval.judge import Judge, JudgeParsingError
from app.schemas.chat import ChatCompletionChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, UsageInfo


class FakeJudgeProvider(BaseLLMProvider):
    """Stand-in judge model: returns a fixed response instead of calling a real LLM."""

    def __init__(self, model_name: str, reply_text: str):
        self._model_name = model_name
        self._reply_text = reply_text
        self.last_prompt = None

    @property
    def name(self) -> str:
        return "fake-judge"

    def get_supported_models(self):
        return {self._model_name: ModelMetadata(model_name=self._model_name, provider_name=self.name)}

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        self.last_prompt = request.messages[0].get_text()
        return ChatCompletionResponse(
            model=self._model_name,
            choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=self._reply_text))],
            usage=UsageInfo(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )

    async def health_check(self) -> bool:
        return True


def _judge_with_fake_reply(reply_text: str) -> tuple[Judge, FakeJudgeProvider]:
    provider = FakeJudgeProvider("fake-judge-model", reply_text)
    registry = ProviderRegistry()
    registry.register_provider(provider)
    router = ModelRouter(registry=registry)
    return Judge(router=router, judge_model="fake-judge-model"), provider


@pytest.mark.asyncio
async def test_judge_evaluate_parses_valid_json_response():
    judge, _ = _judge_with_fake_reply(
        '{"scores": {"relevance": 5, "coherence": 4, "helpfulness": 4}, "rationale": "Clear and on-topic."}'
    )
    result = await judge.evaluate("What is 2+2?", "2+2 equals 4.")
    assert result.scores == {"relevance": 5, "coherence": 4, "helpfulness": 4}
    assert result.overall_score == pytest.approx((5 + 4 + 4) / 3, rel=1e-3)
    assert result.rationale == "Clear and on-topic."


@pytest.mark.asyncio
async def test_judge_evaluate_raises_on_malformed_json():
    judge, _ = _judge_with_fake_reply("this is not json")
    with pytest.raises(JudgeParsingError):
        await judge.evaluate("prompt", "response")


@pytest.mark.asyncio
async def test_judge_evaluate_raises_when_scores_key_missing():
    judge, _ = _judge_with_fake_reply('{"rationale": "no scores field"}')
    with pytest.raises(JudgeParsingError):
        await judge.evaluate("prompt", "response")


@pytest.mark.asyncio
async def test_judge_evaluate_uses_custom_criteria_in_prompt():
    judge, provider = _judge_with_fake_reply('{"scores": {"safety": 5}, "rationale": "Safe."}')
    result = await judge.evaluate("prompt", "response", criteria=["safety"])
    assert "safety" in provider.last_prompt
    assert result.scores == {"safety": 5}


def test_evaluate_endpoint_returns_scores(monkeypatch):
    import app.api.v1.evaluate as evaluate_module
    from app.main import app

    judge, _ = _judge_with_fake_reply('{"scores": {"relevance": 3}, "rationale": "Okay."}')
    monkeypatch.setattr(evaluate_module, "_judge", judge)

    client = TestClient(app)
    response = client.post("/v1/evaluate", json={"prompt": "hi", "response": "hello"})
    assert response.status_code == 200
    data = response.json()
    assert data["scores"] == {"relevance": 3}
    assert data["overall_score"] == 3.0


def test_evaluate_endpoint_returns_502_on_judge_parsing_failure(monkeypatch):
    import app.api.v1.evaluate as evaluate_module

    judge, _ = _judge_with_fake_reply("not valid json")
    monkeypatch.setattr(evaluate_module, "_judge", judge)

    from app.main import app

    client = TestClient(app)
    response = client.post("/v1/evaluate", json={"prompt": "hi", "response": "hello"})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "evaluation_error"


def test_evaluate_endpoint_rejects_empty_prompt(client: TestClient):
    response = client.post("/v1/evaluate", json={"prompt": "", "response": "hello"})
    assert response.status_code == 422
