import os
from dataclasses import dataclass
import math
import json
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    host: str = os.getenv("ANYROUTER_HOST", "127.0.0.1")
    port: int = int(os.getenv("ANYROUTER_PORT", "8000"))
    openai_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    google_key: str | None = os.getenv("GOOGLE_API_KEY")
    openrouter_key: str | None = os.getenv("OPENROUTER_API_KEY")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    connect_timeout: float = float(os.getenv("ANYROUTER_CONNECT_TIMEOUT", "10"))
    first_event_timeout: float = float(os.getenv("ANYROUTER_FIRST_EVENT_TIMEOUT", "30"))
    idle_timeout: float = float(os.getenv("ANYROUTER_IDLE_TIMEOUT", "60"))
    request_timeout: float = float(os.getenv("ANYROUTER_REQUEST_TIMEOUT", "300"))
    queue_timeout: float = float(os.getenv("ANYROUTER_QUEUE_TIMEOUT", "1"))
    max_concurrency: int = int(os.getenv("ANYROUTER_MAX_CONCURRENCY", "8"))
    breaker_threshold: int = int(os.getenv("ANYROUTER_BREAKER_THRESHOLD", "5"))
    breaker_cooldown: float = float(os.getenv("ANYROUTER_BREAKER_COOLDOWN", "30"))
    catalog_refresh: float = float(os.getenv("ANYROUTER_CATALOG_REFRESH", "300"))
    catalog_max_age: float = float(os.getenv("ANYROUTER_CATALOG_MAX_AGE", "900"))
    max_body_bytes: int = int(os.getenv("ANYROUTER_MAX_BODY_BYTES", "1048576"))
    routing_attempts: int = int(os.getenv("ANYROUTER_ROUTING_ATTEMPTS", "2"))
    allow_paid_routing: bool = os.getenv("ANYROUTER_ALLOW_PAID_ROUTING", "false").lower() == "true"
    route_aliases: dict = None
    model_overrides: dict = None
    key_file: str | None = os.getenv("ANYROUTER_KEY_FILE")
    auth_required: bool = os.getenv("ANYROUTER_AUTH_REQUIRED", "false").lower() == "true"
    anonymous_rpm: int = int(os.getenv("ANYROUTER_ANONYMOUS_RPM", "60"))
    max_requests: int = int(os.getenv("ANYROUTER_MAX_REQUESTS", "64"))

    def __post_init__(self):
        for name in ("connect_timeout", "first_event_timeout", "idle_timeout", "request_timeout", "queue_timeout", "breaker_cooldown", "catalog_refresh", "catalog_max_age"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_concurrency <= 0:
            raise ValueError("max_concurrency must be positive")
        for name in ("breaker_threshold", "max_body_bytes", "routing_attempts", "anonymous_rpm", "max_requests"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.routing_attempts > 4:
            raise ValueError("routing_attempts must not exceed 4")
        if self.route_aliases is None:
            self.route_aliases = json.loads(os.getenv("ANYROUTER_ROUTE_ALIASES", "{}"))
        if self.model_overrides is None:
            self.model_overrides = json.loads(os.getenv("ANYROUTER_MODEL_OVERRIDES", "{}"))
        if not isinstance(self.route_aliases, dict) or not isinstance(self.model_overrides, dict):
            raise ValueError("Routing configuration must contain JSON objects")
        for alias, targets in self.route_aliases.items():
            if not isinstance(alias, str) or not alias or alias != alias.strip() or "/" in alias or ":" in alias or not isinstance(targets, list) or not targets or not all(isinstance(t, str) and ":" in t and all(t.split(":", 1)) for t in targets):
                raise ValueError("Aliases require nonempty lists of provider:model targets")
        for target, metadata in self.model_overrides.items():
            if not isinstance(target, str) or ":" not in target or not all(target.split(":", 1)) or not isinstance(metadata, dict) or set(metadata) - {"capabilities", "context", "free"}:
                raise ValueError("Invalid model override")
            capabilities = metadata.get("capabilities", [])
            if not isinstance(capabilities, list) or not all(isinstance(c, str) and c in {"chat", "stream", "tools", "vision", "json_object", "json_schema"} for c in capabilities):
                raise ValueError("Invalid model capabilities")
            context = metadata.get("context")
            if context is not None and (type(context) is not int or context <= 0):
                raise ValueError("Invalid model context")
            if "free" in metadata and type(metadata["free"]) is not bool:
                raise ValueError("Invalid model pricing flag")

    @property
    def has_openai(self):
        return bool(self.openai_key)

    @property
    def has_anthropic(self):
        return bool(self.anthropic_key)

    @property
    def has_google(self):
        return bool(self.google_key)

    @property
    def has_openrouter(self):
        return bool(self.openrouter_key)

cfg = Config()
