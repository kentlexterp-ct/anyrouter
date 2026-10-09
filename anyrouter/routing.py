"""Provider health and conservative routing; explicit targets never fail over."""
from dataclasses import dataclass, field
import ipaddress
import json
import time
from urllib.parse import urlsplit

from .config import cfg
from .errors import GatewayError


class CircuitBreaker:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.state = "closed"
        self.failures = 0
        self.until = 0.0
        self.probe = False
        self.generation = 0
        self.latency = None
        self.failure_rate = 0.0
        self.active = 0
        self.last_error = None

    def usable(self):
        return self.state == "closed" or (self.clock() >= self.until and not self.probe)

    def acquire(self, provider):
        if not self.usable():
            raise GatewayError(503, "circuit_open", "Provider recovery cooldown is active.", provider)
        if self.state != "closed":
            self.state, self.probe = "half_open", True
        self.active += 1
        return self.generation

    def finish(self, ticket, elapsed, error=None):
        self.active -= 1
        if ticket != self.generation:
            return
        was_probe = self.probe
        self.probe = False
        # Cancellations and user/admission errors do not count as provider failures.
        if error is None and elapsed is None:
            if was_probe:
                self.state = "open"
            return
        failure = error is not None and error.code in {
            "provider_error", "provider_timeout", "provider_connect_error",
            "invalid_provider_response", "incomplete_stream", "provider_rate_limit",
        }
        if error and not failure:
            if was_probe:
                self.state = "open"
            return
        self.failure_rate = self.failure_rate * 0.8 + (0.2 if failure else 0)
        if failure:
            self.failures += 1
            self.last_error = error.code
            if was_probe or self.failures >= cfg.breaker_threshold or error.code == "provider_rate_limit":
                self.state = "open"
                self.until = self.clock() + max(cfg.breaker_cooldown, error.retry_after or 0)
                self.generation += 1
        else:
            self.state, self.failures, self.last_error = "closed", 0, None
            self.latency = elapsed if self.latency is None else 0.8 * self.latency + 0.2 * elapsed


def ollama_is_local():
    host = urlsplit(cfg.ollama_url).hostname
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@dataclass(frozen=True)
class ModelRecord:
    provider: str
    model: str
    capabilities: frozenset[str] = field(default_factory=frozenset)
    context: int | None = None
    free: bool | None = None
    updated: float = field(default_factory=time.monotonic)

    @property
    def target(self):
        return f"{self.provider}:{self.model}"


def required_capabilities(body):
    required = {"chat"}
    if body.get("stream"):
        required.add("stream")
    if body.get("tools") or any(m.get("tool_calls") or m.get("role") == "tool" for m in body["messages"]):
        required.add("tools")
    if any(t.get("type") != "function" for t in body.get("tools", [])):
        required.add("custom_tools")  # Unknown extensions are excluded from AUTO.
    if body.get("response_format", {}).get("type") not in (None, "text"):
        required.add(body["response_format"]["type"])
    if any(isinstance(m.get("content"), list) and any(b.get("type") == "image_url" for b in m["content"] if isinstance(b, dict)) for m in body["messages"]):
        required.add("vision")
    return required


class Catalog:
    def __init__(self):
        self.records = {}
        self.discovery = {}

    def replace(self, provider, items):
        records = []
        now = time.monotonic()
        for item in items:
            record = ModelRecord(provider, item) if isinstance(item, str) else ModelRecord(provider=provider, **item)
            if not isinstance(record.model, str) or not record.model:
                raise ValueError("Invalid model ID")
            record = ModelRecord(provider, record.model, frozenset(record.capabilities), record.context, record.free, now)
            override = cfg.model_overrides.get(record.target, {})
            if override:
                capabilities = override.get("capabilities", list(record.capabilities))
                context = override.get("context", record.context)
                free = override.get("free", record.free)
                if not isinstance(capabilities, list) or not all(isinstance(c, str) for c in capabilities) or (context is not None and (type(context) is not int or context <= 0)) or (free is not None and type(free) is not bool):
                    raise ValueError("Invalid model metadata override")
                record = ModelRecord(provider, record.model, frozenset(capabilities), context, free)
            records.append(record)
        replacement = {k: v for k, v in self.records.items() if v.provider != provider}
        replacement.update({r.target: r for r in records})
        self.records = replacement
        self.discovery[provider] = {"ok": True, "updated": now, "error": None}

    def failed(self, provider, code):
        previous = self.discovery.get(provider, {})
        self.discovery[provider] = {"ok": False, "updated": previous.get("updated"), "error": code}

    def candidates(self, alias, body, runtime, allowed=lambda p, m: True):
        targets = cfg.route_aliases.get(alias)
        required = required_capabilities(body)
        routing = body.get("routing", {})
        # Byte count is a conservative text estimate, not a tokenizer guarantee.
        input_size = routing.get("input_tokens") or len(json.dumps({"messages": body["messages"], "tools": body.get("tools"), "response_format": body.get("response_format")}, ensure_ascii=False).encode())
        budget = input_size + (body.get("max_tokens") or 4096)
        result = []
        for record in self.records.values():
            if targets is not None and record.target not in targets:
                continue
            if time.monotonic() - record.updated > cfg.catalog_max_age:
                continue
            if not required <= record.capabilities or record.context is None or record.context < budget:
                continue
            if record.provider == "ollama" and body.get("tool_choice") not in (None, "auto", "none"):
                continue
            if record.provider in ("ollama", "anthropic") and any((t.get("function") or {}).get("strict") for t in body.get("tools", [])):
                continue
            if record.provider == "anthropic" and (
                "json_object" in required or (body.get("temperature") is not None and body["temperature"] > 1)
                or (body.get("temperature") is not None and body.get("top_p") not in (None, 1.0))
            ):
                continue
            if "vision" in required and not routing.get("input_tokens"):
                continue  # Image token cost cannot be inferred from base64 byte length.
            if record.free is None or (not record.free and not cfg.allow_paid_routing):
                continue
            if routing.get("local_only") and not (record.provider == "ollama" and ollama_is_local()):
                continue
            if not allowed(record.provider, record.model):
                continue
            breaker = runtime.breaker(record.provider)
            if breaker.usable() and not runtime.credential_blocked(record.provider):
                result.append(record)
        result.sort(key=lambda r: (
            runtime.breaker(r.provider).failure_rate,
            runtime.breaker(r.provider).active,
            runtime.breaker(r.provider).latency or 0,
            r.target,
        ))
        if not result:
            raise GatewayError(503, "no_eligible_model", "No healthy model satisfies routing policy.")
        return result[:cfg.routing_attempts]
