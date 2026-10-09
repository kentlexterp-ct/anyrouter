"""Small, dependency-free, per-worker telemetry. Never records request payloads."""
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass
import json
import logging
import time


logger = logging.getLogger("anyrouter.requests")


def configure_logging():
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


@dataclass
class RequestTrace:
    request_id: str
    provider: str | None = None
    error: str | None = None


trace = ContextVar("anyrouter_trace", default=None)


class Metrics:
    def __init__(self):
        self.requests = Counter()
        self.upstream = Counter()
        self.duration = Counter()
        self.tokens = Counter()

    def render(self, runtime):
        lines = ["# TYPE anyrouter_requests_total counter"]
        for (route, status), value in sorted(self.requests.items()):
            lines.append(f'anyrouter_requests_total{{route="{route}",status="{status}"}} {value}')
        lines.append("# TYPE anyrouter_upstream_total counter")
        for (provider, outcome), value in sorted(self.upstream.items()):
            lines.append(f'anyrouter_upstream_total{{provider="{provider}",outcome="{outcome}"}} {value}')
        for provider, value in sorted(self.duration.items()):
            lines.append(f'anyrouter_upstream_seconds_sum{{provider="{provider}"}} {value}')
        for provider, value in sorted(self.tokens.items()):
            lines.append(f'anyrouter_tokens_total{{provider="{provider}"}} {value}')
        for provider, breaker in sorted(runtime.breakers.items()):
            for state in ("closed", "open", "half_open"):
                lines.append(f'anyrouter_circuit_state{{provider="{provider}",state="{state}"}} {int(breaker.state == state)}')
        return "\n".join(lines) + "\n"


metrics = Metrics()


def upstream_result(provider, started, error=None):
    current = trace.get()
    if current:
        current.provider = provider
        if error:
            current.error = error.code
    elapsed = time.monotonic() - started
    metrics.upstream[provider, "error" if error else "success"] += 1
    metrics.duration[provider] += elapsed
    return elapsed


def log_request(request_id, route, status, elapsed, current):
    metrics.requests[route, status] += 1
    logger.info(json.dumps({
        "event": "request_completed", "request_id": request_id, "route": route,
        "status": status, "duration_ms": round(elapsed * 1000, 2),
        "provider": current.provider, "error": current.error,
    }, separators=(",", ":")))
