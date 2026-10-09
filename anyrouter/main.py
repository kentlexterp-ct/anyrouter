from contextlib import asynccontextmanager

import json
import math
import asyncio
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import cfg
from .errors import GatewayError, provider_error
from . import providers as prov_mod
from . import transport
from .streaming import ManagedStreamingResponse, until_disconnect
from .security import GatewayMiddleware, keys, allows, request_policy, loopback
from .observability import metrics, trace


_MODEL_INDEX = {}


@asynccontextmanager
async def lifespan(app):
    keys.load()
    if (cfg.auth_required or not loopback(cfg.host)) and not keys.keys:
        raise RuntimeError("Caller API keys are required for authenticated or public deployment")
    async with transport.lifespan() as runtime:
        async def refresh_provider(name, provider):
            try:
                discover = getattr(provider, "list_catalog", provider.list_models)
                runtime.catalog.replace(name, await runtime.call(name, discover))
            except Exception as exc:
                runtime.catalog.failed(name, provider_error(name, exc).code)

        async def refresh():
            configured = prov_mod.all_providers()
            await asyncio.gather(*(refresh_provider(name, provider) for name, provider in configured.items()))
            # A stale catalog is retained for diagnostics, but cannot route automatically.
            runtime.catalog.records = {k: r for k, r in runtime.catalog.records.items() if r.provider in configured}
            _MODEL_INDEX.clear()
            for record in sorted(runtime.catalog.records.values(), key=lambda r: r.target):
                _MODEL_INDEX.setdefault(record.model, record.provider)

        await refresh()
        async def refresh_loop():
            while True:
                await asyncio.sleep(cfg.catalog_refresh)
                await refresh()
        task = asyncio.create_task(refresh_loop())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


app = FastAPI(title="anyrouter", lifespan=lifespan)
app.add_middleware(GatewayMiddleware)


@app.exception_handler(GatewayError)
async def gateway_error_handler(request, exc):
    if trace.get():
        trace.get().error = exc.code
    headers = {"Retry-After": str(int(exc.retry_after))} if exc.retry_after else {}
    return JSONResponse(exc.payload(), status_code=exc.status_code, headers=headers)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request, exc):
    if trace.get():
        trace.get().error = "invalid_request"
    error = GatewayError(422, "invalid_request", "Invalid request body.").payload()
    error["error"]["details"] = [
        {"location": list(item["loc"]), "type": item["type"]}
        for item in exc.errors()
    ]
    return JSONResponse(error, status_code=422)


class RoutingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    local_only: bool = False
    input_tokens: int | None = Field(default=None, gt=0, strict=True)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[dict] = Field(min_length=1)
    temperature: float | None = Field(default=1.0, ge=0, allow_inf_nan=False)
    top_p: float | None = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    max_tokens: int | None = Field(default=None, gt=0, strict=True)
    stream: bool = False
    stop: list[str] | str | None = None
    tools: list[dict] | None = None
    tool_choice: dict | str | None = None
    response_format: dict | None = None
    stream_options: StreamOptions | None = None
    routing: RoutingOptions | None = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("model must not be blank")
        return value

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, messages):
        for message in messages:
            if not isinstance(message.get("role"), str) or message["role"] not in {"system", "developer", "user", "assistant", "tool"}:
                raise ValueError("unsupported message role")
            content = message.get("content")
            if not isinstance(content, (str, list)):
                if not (message["role"] == "assistant" and message.get("tool_calls")):
                    raise ValueError("each message requires content or assistant tool calls")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict) or not isinstance(block.get("type"), str):
                        raise ValueError("content blocks require a type")
                    if block["type"] == "text" and not isinstance(block.get("text"), str):
                        raise ValueError("text blocks require text")
            if message["role"] == "tool" and (not isinstance(message.get("tool_call_id"), str) or not message["tool_call_id"]):
                raise ValueError("tool messages require a call ID")
            calls = message.get("tool_calls")
            if calls is not None:
                if message["role"] != "assistant" or not isinstance(calls, list):
                    raise ValueError("tool calls require an assistant message")
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    if not isinstance(function, dict) or not isinstance(call.get("id"), str) or call.get("type") != "function" or not isinstance(function.get("name"), str) or not isinstance(function.get("arguments"), str):
                        raise ValueError("invalid function call")
        return messages

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, value):
        for tool in value or []:
            if tool.get("type") == "function":
                function = tool.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"] or not isinstance(function.get("parameters", {}), dict):
                    raise ValueError("invalid function definition")
        return value

    @field_validator("tool_choice")
    @classmethod
    def validate_tool_choice(cls, value):
        if isinstance(value, str) and value not in ("auto", "none", "required"):
            raise ValueError("invalid tool choice")
        if isinstance(value, dict):
            function = value.get("function")
            if value.get("type") != "function" or not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
                raise ValueError("invalid named tool choice")
        return value

    @field_validator("response_format")
    @classmethod
    def validate_format(cls, value):
        if value is None:
            return value
        kind = value.get("type")
        if kind not in ("text", "json_object", "json_schema"):
            raise ValueError("invalid response format")
        if kind == "json_schema":
            schema = value.get("json_schema")
            if not isinstance(schema, dict) or not isinstance(schema.get("name"), str) or not schema["name"] or not isinstance(schema.get("schema"), dict):
                raise ValueError("json_schema requires name and schema")
        return value

    @model_validator(mode="after")
    def validate_stream_options(self):
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires streaming")
        return self


def resolve(model):
    # Colon syntax avoids collisions with native OpenRouter IDs such as openai/....
    prefix, separator, tail = model.partition(":")
    if separator and prov_mod.get(prefix):
        if not tail:
            raise GatewayError(422, "invalid_request", "Provider model must not be empty.")
        return prefix, tail
    if "/" in model:
        p, m = model.split("/", 1)
        if prov_mod.get(p):
            if not m:
                raise GatewayError(422, "invalid_request", "Provider model must not be empty.")
            return p, m
    if model in _MODEL_INDEX:
        return _MODEL_INDEX[model], model
    configured = prov_mod.all_providers()
    # Preserve the local/single-provider workflow; multi-provider ambiguity fails closed.
    if len(configured) == 1:
        return next(iter(configured)), model
    if configured:
        raise GatewayError(404, "unknown_model", "Unknown model; use an explicit provider prefix.")
    raise GatewayError(502, "no_provider", "No provider configured.")


@app.get("/v1/models")
async def models(request: Request = None):
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "anyrouter", "provider": p}
            for m, p in _MODEL_INDEX.items()
            if allows(request_policy(request), p, m)
        ],
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest, request: Request = None):
    body = req.model_dump(exclude_none=True, exclude_unset=True)
    policy = request_policy(request)
    runtime = transport.current_runtime()
    automatic = req.model == "auto" or req.model in cfg.route_aliases
    if automatic:
        body.setdefault("max_tokens", 4096)
        candidates = [(r.provider, r.model) for r in runtime.catalog.candidates(req.model, body, runtime, lambda p, m: allows(policy, p, m))]
    else:
        candidates = [resolve(req.model)]
    deadline = asyncio.get_running_loop().time() + cfg.request_timeout
    for index, (prov_name, actual) in enumerate(candidates):
        if not allows(policy, prov_name, actual):
            raise GatewayError(403, "forbidden", "API key policy does not permit this model.")
        if body.get("routing", {}).get("local_only"):
            from .routing import ollama_is_local
            if prov_name != "ollama" or not ollama_is_local():
                raise GatewayError(403, "privacy_policy", "Routing requires a local provider.")
        try:
            return await dispatch(req, request, body, prov_name, actual, runtime, deadline, automatic)
        except GatewayError as exc:
            # Only known pre-generation failures are safe to send to another model.
            if not automatic or index == len(candidates) - 1 or exc.code not in {"circuit_open", "provider_busy", "provider_connect_error", "provider_rate_limit"}:
                raise


async def dispatch(req, request, body, prov_name, actual, runtime, deadline, automatic=False):
    p = prov_mod.get(prov_name)
    if not p or not p.available:
        raise GatewayError(502, "provider_unavailable", "Provider unavailable.", prov_name)
    if prov_name == "ollama":
        body = dict(body)
        for field in ("temperature", "top_p"):
            if field not in req.model_fields_set:
                body[field] = getattr(req, field)

    if req.stream:
        stream = runtime.stream(prov_name, p.stream_chat(actual, {**body, "_automatic": automatic}), deadline)
        try:
            first = await until_disconnect(anext(stream), request)
        except StopAsyncIteration:
            raise GatewayError(502, "invalid_provider_response", "Provider returned an empty stream.", prov_name)
        except BaseException as exc:
            await stream.aclose()
            if isinstance(exc, Exception):
                raise provider_error(prov_name, exc) from None
            raise

        async def gen():
            try:
                yield first
                async for c in stream:
                    yield c
            except Exception as exc:
                error = provider_error(prov_name, exc).payload()
                yield f"data: {json.dumps(error)}\n\n".encode()
            finally:
                await stream.aclose()
        return ManagedStreamingResponse(gen(), stream, deadline, prov_name)

    try:
        resp = await until_disconnect(
            runtime.call(prov_name, lambda: p.chat(actual, {**body, "_automatic": automatic}), deadline), request,
        )
    except Exception as exc:
        raise provider_error(prov_name, exc) from None
    return JSONResponse(resp)


@app.post("/webhook/classify")
async def classify_lead(request: Request):
    try:
        payload = await request.json()
    except (ValueError, UnicodeDecodeError):
        raise GatewayError(400, "invalid_json", "Request body must contain valid JSON.") from None

    if not isinstance(payload, dict):
        raise GatewayError(422, "invalid_request", "Lead payload must be a JSON object.")

    lead_fields = {
        "first_name", "last_name", "email", "phone", "source", "lead_source", "utm_source",
        "service", "interest", "product", "message", "notes", "form_message", "comments",
        "budget", "timeline", "company", "company_name",
    }
    for key in lead_fields & payload.keys():
        value = payload[key]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (str, int, float))
            or (isinstance(value, float) and not math.isfinite(value))
        ):
            raise GatewayError(422, "invalid_request", "Lead fields must contain text, numbers, or null.")

    def get(*keys):
        for k in keys:
            v = payload.get(k)
            if v is not None:
                value = str(v).strip()
                if value:
                    return value
        return ""

    first   = get("first_name")
    last    = get("last_name")
    email   = get("email")
    phone   = get("phone")
    source  = get("source", "lead_source", "utm_source")
    service = get("service", "interest", "product")
    message = get("message", "notes", "form_message", "comments")
    budget  = get("budget", "timeline")
    company = get("company", "company_name")

    name = f"{first} {last}".strip() or "Unknown"

    brief_lines = [f"Name: {name}"]
    if email:    brief_lines.append(f"Email: {email}")
    if phone:    brief_lines.append(f"Phone: {phone}")
    if company:  brief_lines.append(f"Company: {company}")
    if source:   brief_lines.append(f"Lead source: {source}")
    if service:  brief_lines.append(f"Interested in: {service}")
    if budget:   brief_lines.append(f"Budget / timeline: {budget}")
    if message:  brief_lines.append(f"Message from lead: {message}")

    has_signal = any([source, service, message, budget, company, phone])
    response = {
        "classification": "warm", "raw": "", "name": name,
        "signals_present": has_signal, "brief": "\n".join(brief_lines), "fallback": False,
    }

    def fallback(error):
        headers = {"Retry-After": str(int(error.retry_after))} if error.retry_after else {}
        return JSONResponse(
            {**response, "fallback": True, **error.payload()}, status_code=error.status_code,
            headers=headers,
        )

    system_prompt = (
        "You are a strict lead qualification classifier. Reply with EXACTLY "
        "one word: hot, warm, or cold.\n\n"
        "AUTOMATIC COLD if any of these appear in the lead's message:\n"
        "- 'just looking', 'looking around', 'browsing', 'no rush', "
        "'someday', 'maybe later', 'curious', 'checking out', 'saw your site', "
        "'random', 'not sure yet'\n"
        "Or if the email domain is generic AND the message is vague.\n"
        "Or if the source is blog/organic AND there is no buying signal.\n\n"
        "AUTOMATIC HOT if any of these appear:\n"
        "- 'pricing', 'quote', 'buy', 'purchase', 'start this week', "
        "'start this month', 'urgent', 'asap', 'call me', 'ready to', "
        "'need this', 'timeline', specific seat counts or budget numbers.\n\n"
        "WARM only if there is genuine interest with no urgency — "
        "questions, comparisons, 'interested' with no timeline, "
        "'learn more' but with specific intent.\n\n"
        "If ONLY a name and email (no other data): reply 'warm'.\n"
        "If unsure between hot and warm: reply 'warm'.\n"
        "If unsure between warm and cold: reply 'cold'.\n\n"
        "Reply with one word only. No punctuation. No explanation."
    )

    ollama = prov_mod.get("ollama")
    if not allows(request_policy(request), "ollama", "llama3.2:latest"):
        raise GatewayError(403, "forbidden", "API key policy does not permit the classifier model.")
    if not ollama or not ollama.available:
        return fallback(GatewayError(503, "provider_unavailable", "Classifier unavailable.", "ollama"))

    chat_body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(brief_lines)},
        ],
        "temperature": 0.1,
        "max_tokens": 10,
    }

    try:
        result = await until_disconnect(
            transport.current_runtime().call("ollama", lambda: ollama.chat("llama3.2:latest", chat_body)), request,
        )
    except Exception as exc:
        return fallback(provider_error("ollama", exc))

    try:
        raw = result["choices"][0]["message"]["content"].strip().lower()
    except (KeyError, IndexError, TypeError, AttributeError):
        return fallback(GatewayError(502, "invalid_provider_response", "Invalid classifier response.", "ollama"))
    if raw not in {"hot", "warm", "cold"}:
        return fallback(GatewayError(502, "invalid_classification", "Classifier must return hot, warm, or cold.", "ollama"))

    return {**response, "classification": raw, "raw": raw}


@app.get("/")
async def root():
    return {
        "anyrouter": "up",
        "models": len(_MODEL_INDEX),
        "providers": list(prov_mod.all_providers().keys()),
    }


@app.get("/ready")
async def ready():
    runtime = transport.current_runtime()
    states = {}
    for name in prov_mod.all_providers():
        discovery = runtime.catalog.discovery.get(name, {})
        updated = discovery.get("updated")
        healthy = bool(updated is not None and discovery.get("ok") and time.monotonic() - updated <= cfg.catalog_max_age and runtime.breaker(name).usable() and not runtime.credential_blocked(name))
        states[name] = {"configured": True, "ready": healthy, "circuit": runtime.breaker(name).state}
    available = any(p["ready"] for p in states.values())
    return JSONResponse({"ready": available, "providers": states}, status_code=200 if available else 503)


@app.get("/metrics")
async def prometheus_metrics():
    return PlainTextResponse(metrics.render(transport.current_runtime()), media_type="text/plain; version=0.0.4")


def run():
    import uvicorn
    from .observability import configure_logging
    configure_logging()
    uvicorn.run("anyrouter.main:app", host=cfg.host, port=cfg.port, access_log=False)


if __name__ == "__main__":
    run()
