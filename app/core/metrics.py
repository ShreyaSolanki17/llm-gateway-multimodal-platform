from collections import defaultdict
from typing import Dict, List


class MetricsRegistry:
    """In-memory metrics store: request counts, cost/latency, cache and error breakdowns.

    ponytail: per-process in-memory counters, reset on restart, not shared
    across replicas -- fine at prototype scale, swap for a real metrics
    backend (Prometheus + pushgateway, or a shared store) if this ever runs
    with multiple workers/instances.
    """

    def __init__(self):
        self.request_count = 0
        self.requests_by_model: Dict[str, int] = defaultdict(int)
        self.errors_by_type: Dict[str, int] = defaultdict(int)
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_cost_usd = 0.0
        self._latencies_ms: List[float] = []

    def record_request(self, model: str, latency_ms: float, cost_usd: float, cache_hit: bool) -> None:
        self.request_count += 1
        self.requests_by_model[model] += 1
        self.total_cost_usd += cost_usd
        self._latencies_ms.append(latency_ms)
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

    def record_error(self, error_type: str) -> None:
        self.errors_by_type[error_type] += 1

    def average_latency_ms(self) -> float:
        if not self._latencies_ms:
            return 0.0
        return sum(self._latencies_ms) / len(self._latencies_ms)

    def snapshot(self) -> dict:
        return {
            "request_count": self.request_count,
            "requests_by_model": dict(self.requests_by_model),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": round(self.cache_hits / self.request_count, 4) if self.request_count else 0.0,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "average_latency_ms": round(self.average_latency_ms(), 2),
            "errors_by_type": dict(self.errors_by_type),
        }

    def as_prometheus_text(self) -> str:
        snap = self.snapshot()
        lines = [
            "# HELP gateway_requests_total Total requests served",
            "# TYPE gateway_requests_total counter",
            f"gateway_requests_total {snap['request_count']}",
            "# HELP gateway_cache_hits_total Semantic cache hits",
            "# TYPE gateway_cache_hits_total counter",
            f"gateway_cache_hits_total {snap['cache_hits']}",
            "# HELP gateway_cache_misses_total Semantic cache misses",
            "# TYPE gateway_cache_misses_total counter",
            f"gateway_cache_misses_total {snap['cache_misses']}",
            "# HELP gateway_cost_usd_total Cumulative estimated cost in USD",
            "# TYPE gateway_cost_usd_total counter",
            f"gateway_cost_usd_total {snap['total_cost_usd']}",
            "# HELP gateway_latency_ms_avg Average request latency in milliseconds",
            "# TYPE gateway_latency_ms_avg gauge",
            f"gateway_latency_ms_avg {snap['average_latency_ms']}",
        ]
        for model, count in snap["requests_by_model"].items():
            lines.append(f'gateway_requests_by_model_total{{model="{model}"}} {count}')
        for error_type, count in snap["errors_by_type"].items():
            lines.append(f'gateway_errors_total{{type="{error_type}"}} {count}')
        return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()
