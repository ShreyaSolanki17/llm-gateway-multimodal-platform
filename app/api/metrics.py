from fastapi import APIRouter, Response
from app.core.metrics import metrics_registry

router = APIRouter(tags=["Observability"])


@router.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus-format metrics: request counts, cost, latency, cache and error breakdowns."""
    return Response(content=metrics_registry.as_prometheus_text(), media_type="text/plain; version=0.0.4; charset=utf-8")
