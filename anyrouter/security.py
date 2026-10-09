"""Caller keys and admission limits. Upstream credentials never enter this store."""
import hashlib
import hmac
import ipaddress
import json
from pathlib import Path
import time
import uuid
import asyncio

from starlette.responses import JSONResponse

from .config import cfg
from .errors import GatewayError
from .observability import RequestTrace, trace, log_request


SCOPES = {"chat", "models", "classify", "admin"}
PATH_SCOPES = {"/v1/chat/completions": "chat", "/v1/models": "models", "/webhook/classify": "classify", "/metrics": "admin"}


def loopback(host):
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class KeyStore:
    def __init__(self):
        self.keys = {}
        self.source = None
        self.windows = {}

    def load(self):
        if not cfg.key_file:
            self.keys, self.source = {}, None
            return
        try:
            # Read afresh so rotation/revocation applies to the next admission.
            with Path(cfg.key_file).open("rb") as handle:
                content = handle.read(1048577)
            if len(content) > 1048576:
                raise ValueError("Key file too large")
            fingerprint = hashlib.sha256(content).digest()
            if self.source == (cfg.key_file, fingerprint):
                return
            keys = json.loads(content)
            if not isinstance(keys, dict) or len(keys) > 1000:
                raise ValueError("Invalid key store")
            digests = set()
            for key_id, policy in keys.items():
                if not isinstance(key_id, str) or not isinstance(policy, dict):
                    raise ValueError("Invalid key policy")
                digest = policy.get("sha256", "")
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("Keys must use lowercase SHA-256 digests")
                if digest in digests or set(policy) - {"sha256", "scopes", "providers", "models", "rpm", "daily_requests", "disabled", "local_only", "expires_at"}:
                    raise ValueError("Duplicate digest or unsupported key policy field")
                digests.add(digest)
                scopes = policy.get("scopes", [])
                if not isinstance(scopes, list) or not all(s in SCOPES for s in scopes):
                    raise ValueError("Invalid key scopes")
                for field in ("providers", "models"):
                    values = policy.get(field)
                    if values is not None and (not isinstance(values, list) or not all(isinstance(v, str) for v in values)):
                        raise ValueError("Invalid allowlist")
                for field in ("rpm", "daily_requests"):
                    value = policy.get(field, 60 if field == "rpm" else 1000)
                    if type(value) is not int or value <= 0:
                        raise ValueError("Invalid quota")
                expiry = policy.get("expires_at")
                if expiry is not None and (type(expiry) not in (int, float) or not 0 < expiry < float("inf")):
                    raise ValueError("Invalid expiry")
                if type(policy.get("disabled", False)) is not bool or type(policy.get("local_only", False)) is not bool:
                    raise ValueError("Invalid key flags")
            self.keys, self.source = keys, (cfg.key_file, fingerprint)
            # Keep counters for keys surviving rotation; bound memory after removals.
            self.windows = {k: v for k, v in self.windows.items() if k in keys or k == "anonymous"}
        except (OSError, ValueError, TypeError):
            raise GatewayError(503, "key_store_unavailable", "Caller key configuration is unavailable.") from None

    def authenticate(self, scope, authorization, remote):
        self.load()
        required = cfg.auth_required or cfg.key_file or not loopback(cfg.host) or not loopback(remote)
        if not authorization and not required and scope != "admin":
            return "anonymous", {"scopes": ["chat", "models", "classify"], "rpm": cfg.anonymous_rpm, "daily_requests": 10000}
        if not authorization or not authorization.startswith("Bearer "):
            raise GatewayError(401, "unauthorized", "A valid bearer API key is required.")
        raw = authorization[7:]
        if not raw or len(raw) > 4096:
            raise GatewayError(401, "unauthorized", "A valid bearer API key is required.")
        digest = hashlib.sha256(raw.encode()).hexdigest()
        match = None
        for key_id, policy in self.keys.items():
            if hmac.compare_digest(digest, policy["sha256"]):
                match = key_id, policy
        if match is None or match[1].get("disabled") or match[1].get("expires_at", float("inf")) <= time.time():
            raise GatewayError(401, "unauthorized", "A valid bearer API key is required.")
        if scope not in match[1].get("scopes", []):
            raise GatewayError(403, "forbidden", "API key scope does not permit this route.")
        return match

    def admit(self, key_id, policy):
        minute, day = int(time.monotonic() // 60), int(time.time() // 86400)
        previous = self.windows.get(key_id, (minute, 0, day, 0))
        count = previous[1] if previous[0] == minute else 0
        daily = previous[3] if previous[2] == day else 0
        if count >= policy.get("rpm", 60):
            raise GatewayError(429, "rate_limit", "Caller request rate limit reached.", retry_after=60)
        if daily >= policy.get("daily_requests", 1000):
            raise GatewayError(429, "quota_exceeded", "Caller daily request quota reached.")
        self.windows[key_id] = (minute, count + 1, day, daily + 1)


keys = KeyStore()


def allows(policy, provider, model):
    from .routing import ollama_is_local
    return (
        (policy.get("providers") is None or provider in policy["providers"])
        and (policy.get("models") is None or f"{provider}:{model}" in policy["models"])
        and (not policy.get("local_only") or (provider == "ollama" and ollama_is_local()))
    )


def request_policy(request):
    return request.scope.get("state", {}).get("key_policy", {}) if request else {}


class GatewayMiddleware:
    def __init__(self, app):
        self.app = app
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request_id = uuid.uuid4().hex
        current = RequestTrace(request_id)
        token = trace.set(current)
        started, status = time.monotonic(), 500
        admitted = False
        response_started = False
        path = scope.get("path", "")
        root_path = scope.get("root_path", "")
        if root_path and path.startswith(root_path + "/"):
            path = path[len(root_path):]
        route = path if path in PATH_SCOPES or path in ("/", "/ready") else "other"
        async def traced_send(message):
            nonlocal status, response_started
            if message["type"] == "http.response.start":
                status = message["status"]
                response_started = True
                message = {**message, "headers": [*message.get("headers", []), (b"x-request-id", request_id.encode())]}
            await send(message)
        try:
            if self.active >= cfg.max_requests:
                raise GatewayError(503, "gateway_busy", "Gateway request limit reached.")
            self.active += 1
            admitted = True
            route_scope = PATH_SCOPES.get(path)
            if route_scope:
                headers = scope.get("headers", [])
                auth = [v for k, v in headers if k.lower() == b"authorization"]
                if len(auth) > 1:
                    raise GatewayError(401, "unauthorized", "A single bearer API key is required.")
                try:
                    authorization = auth[0].decode("ascii") if auth else None
                except UnicodeDecodeError:
                    raise GatewayError(401, "unauthorized", "A valid bearer API key is required.") from None
                key_id, policy = keys.authenticate(route_scope, authorization, (scope.get("client") or ("",))[0])
                keys.admit(key_id, policy)
                scope.setdefault("state", {})["key_policy"] = policy
            body_size = 0
            async def bounded_receive():
                nonlocal body_size
                message = await receive()
                if message["type"] == "http.request":
                    body_size += len(message.get("body", b""))
                    if body_size > cfg.max_body_bytes:
                        raise GatewayError(413, "request_too_large", "Request body exceeds the configured limit.")
                return message
            # Buffer only request bodies, before FastAPI parsing, for consistent 413 errors.
            if scope.get("method") in ("POST", "PUT", "PATCH"):
                messages = []
                deadline = asyncio.get_running_loop().time() + cfg.request_timeout
                while True:
                    try:
                        message = await asyncio.wait_for(bounded_receive(), max(0, deadline - asyncio.get_running_loop().time()))
                    except asyncio.TimeoutError:
                        raise GatewayError(408, "request_timeout", "Request body deadline exceeded.") from None
                    if message["type"] == "http.disconnect":
                        raise GatewayError(499, "client_disconnected", "Client disconnected.")
                    messages.append(message)
                    if not message.get("more_body", False):
                        break
                async def replay():
                    return messages.pop(0) if messages else await receive()
                await self.app(scope, replay, traced_send)
            else:
                await self.app(scope, bounded_receive, traced_send)
        except GatewayError as exc:
            current.error = exc.code
            if response_started:
                raise  # Never send a second HTTP response after headers are committed.
            headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else {}
            await JSONResponse(exc.payload(), status_code=exc.status_code, headers=headers)(scope, receive, traced_send)
        finally:
            if admitted:
                self.active -= 1
            log_request(request_id, route, status, time.monotonic() - started, current)
            trace.reset(token)
