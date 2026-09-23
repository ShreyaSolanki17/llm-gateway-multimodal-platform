from fastapi.testclient import TestClient
from app.core.metrics import MetricsRegistry


def test_record_request_updates_counts_and_cost():
    registry = MetricsRegistry()
    registry.record_request("mock-gpt-4o", latency_ms=100.0, cost_usd=0.01, cache_hit=False)
    registry.record_request("mock-gpt-4o", latency_ms=200.0, cost_usd=0.02, cache_hit=True)

    snap = registry.snapshot()
    assert snap["request_count"] == 2
    assert snap["requests_by_model"] == {"mock-gpt-4o": 2}
    assert snap["cache_hits"] == 1
    assert snap["cache_misses"] == 1
    assert snap["cache_hit_rate"] == 0.5
    assert snap["total_cost_usd"] == 0.03
    assert snap["average_latency_ms"] == 150.0


def test_record_error_updates_error_breakdown():
    registry = MetricsRegistry()
    registry.record_error("authentication_error")
    registry.record_error("authentication_error")
    registry.record_error("rate_limit_exceeded")

    snap = registry.snapshot()
    assert snap["errors_by_type"] == {"authentication_error": 2, "rate_limit_exceeded": 1}


def test_snapshot_on_empty_registry_has_no_division_by_zero():
    registry = MetricsRegistry()
    snap = registry.snapshot()
    assert snap["request_count"] == 0
    assert snap["cache_hit_rate"] == 0.0
    assert snap["average_latency_ms"] == 0.0


def test_as_prometheus_text_includes_labeled_metrics():
    registry = MetricsRegistry()
    registry.record_request("gpt-4o", latency_ms=50.0, cost_usd=0.005, cache_hit=False)
    registry.record_error("validation_error")

    text = registry.as_prometheus_text()
    assert "gateway_requests_total 1" in text
    assert 'gateway_requests_by_model_total{model="gpt-4o"} 1' in text
    assert 'gateway_errors_total{type="validation_error"} 1' in text


def test_metrics_endpoint_reflects_chat_activity(client: TestClient):
    client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "gateway_requests_total" in response.text


def test_metrics_endpoint_records_validation_errors(client: TestClient):
    client.post("/v1/chat/completions", json={"model": "mock-gpt-4o", "messages": []})
    response = client.get("/metrics")
    assert 'gateway_errors_total{type="validation_error"}' in response.text


def test_metrics_endpoint_open_without_auth(monkeypatch):
    from app.config import settings
    from app.main import app

    monkeypatch.setattr(settings, "GATEWAY_API_KEY", "secret-key")
    with TestClient(app) as authed_client:
        response = authed_client.get("/metrics")
    assert response.status_code == 200
