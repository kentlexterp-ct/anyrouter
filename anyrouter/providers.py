import json
import httpx
import asyncio
import re
from decimal import Decimal, InvalidOperation
from .compat import (anthropic_request, anthropic_response, anthropic_reason, ollama_request,
                     ollama_response, ollama_calls, invalid_response, arguments)
from .compat import validate_completion, Provider
from .config import cfg
from .errors import GatewayError
from .transport import client_for, http_timeout
from .streaming import ChunkEncoder, bounded_lines, event_json, passthrough_sse, sse_events, usage_counts


def reject_unsupported(body, provider, fields):
    if any(body.get(field) is not None for field in fields):
        raise GatewayError(422, "unsupported_parameter", "Parameter unsupported by this adapter.", provider)


class OpenAIProvider:
    name = "openai"
    base = "https://api.openai.com/v1"

    @property
    def available(self):
        return cfg.has_openai

    def headers(self):
        return {
            "Authorization": f"Bearer {cfg.openai_key}",
            "Content-Type": "application/json",
        }

    async def list_models(self):
        async with client_for(self.name) as c:
            r = await c.get(f"{self.base}/models", headers=self.headers(), timeout=http_timeout(15))
            r.raise_for_status()
            return [m["id"] for m in r.json()["data"]]

    async def chat(self, model, body):
        p = {k: v for k, v in body.items()
             if k in ("messages","temperature","top_p","max_tokens","stop","tools","tool_choice","response_format")}
        p["model"] = model
        async with client_for(self.name) as c:
            r = await c.post(f"{self.base}/chat/completions", headers=self.headers(), json=p)
            r.raise_for_status()
            return validate_completion(r.json(), self.name)

    async def stream_chat(self, model, body):
        p = {k: v for k, v in body.items()
             if k in ("messages","temperature","top_p","max_tokens","stop","tools","tool_choice","response_format","stream_options")}
        p["model"] = model
        p["stream"] = True
        async with client_for(self.name) as c:
            async with c.stream("POST", f"{self.base}/chat/completions", headers=self.headers(), json=p) as r:
                r.raise_for_status()
                async for chunk in passthrough_sse(r, self.name):
                    yield chunk


class AnthropicProvider:
    name = "anthropic"
    base = "https://api.anthropic.com/v1"

    @property
    def available(self):
        return cfg.has_anthropic

    def headers(self):
        return {
            "x-api-key": cfg.anthropic_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    async def list_models(self):
        return [item["model"] for item in await self.list_catalog()]

    async def list_catalog(self):
        result, cursor, seen = [], None, set()
        async with client_for(self.name) as client:
            for _ in range(20):
                params = {"limit": 1000}
                if cursor:
                    params["after_id"] = cursor
                response = await client.get(f"{self.base}/models", params=params, headers=self.headers(), timeout=http_timeout(15))
                response.raise_for_status()
                page = response.json()
                for item in page["data"]:
                    caps = {"chat", "stream"}
                    metadata = item.get("capabilities") or {}
                    if (metadata.get("image_input") or {}).get("supported") is True:
                        caps.add("vision")
                    if (metadata.get("structured_outputs") or {}).get("supported") is True:
                        caps.add("json_schema")
                    # The model-list API does not declare client-tool support.
                    result.append({"model": item["id"], "capabilities": frozenset(caps), "context": positive_int(item.get("max_input_tokens")), "free": False})
                if not page.get("has_more"):
                    return result
                cursor = page.get("last_id")
                if not cursor or cursor in seen:
                    invalid_response(self.name)
                seen.add(cursor)
        invalid_response(self.name)

    def _translate(self, body):
        return anthropic_request(body)

    def _wrap(self, resp):
        return anthropic_response(resp)

    async def chat(self, model, body):
        p = self._translate({**body, "model": model})
        async with client_for(self.name) as c:
            r = await c.post(f"{self.base}/messages", headers=self.headers(), json=p)
            r.raise_for_status()
            return self._wrap(r.json())

    async def stream_chat(self, model, body):
        p = self._translate({**body, "model": model}); p["stream"] = True
        encoder = ChunkEncoder(model)
        prompt_tokens, completion_tokens, reason = None, None, "stop"
        tool_blocks = {}
        argument_bytes = 0
        cache_tokens = 0
        async with client_for(self.name) as c:
            async with c.stream("POST", f"{self.base}/messages", headers=self.headers(), json=p) as r:
                r.raise_for_status()
                async for data, _ in sse_events(r, self.name):
                    ev = event_json(data, self.name)
                    if ev.get("type") == "message_start":
                        message = ev.get("message", {})
                        if not isinstance(message, dict):
                            raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", self.name)
                        encoder.id = message.get("id") or encoder.id
                        encoder.model = message.get("model") or model
                        usage = message.get("usage") or {}
                        if not isinstance(usage, dict):
                            raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", self.name)
                        prompt_tokens = usage.get("input_tokens")
                        completion_tokens = usage.get("output_tokens")
                        for field in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                            value = usage.get(field, 0)
                            if type(value) is not int or value < 0:
                                invalid_response(self.name)
                            cache_tokens += value
                        yield encoder.encode({"role": "assistant"})
                    if ev.get("type") == "message_delta":
                        delta = ev.get("delta") or {}
                        if not isinstance(delta, dict):
                            raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", self.name)
                        stop = delta.get("stop_reason")
                        if stop:
                            reason = anthropic_reason(stop)
                        usage = ev.get("usage") or {}
                        if not isinstance(usage, dict):
                            raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", self.name)
                        prompt_tokens = usage.get("input_tokens", prompt_tokens)
                        completion_tokens = usage.get("output_tokens", completion_tokens)
                    if ev.get("type") == "message_stop":
                        if any(not block["closed"] for block in tool_blocks.values()):
                            invalid_response(self.name)
                        prompt = prompt_tokens + cache_tokens if type(prompt_tokens) is int else None
                        yield terminal_chunk(encoder, reason, usage_counts(prompt, completion_tokens), body, self.name)
                        yield b"data: [DONE]\n\n"
                        return
                    if ev.get("type") == "content_block_start":
                        block = ev.get("content_block")
                        if not isinstance(block, dict):
                            invalid_response(self.name)
                        if block.get("type") == "tool_use":
                            index = ev.get("index")
                            if type(index) is not int or index < 0 or index in tool_blocks or len(tool_blocks) >= 128 or not isinstance(block.get("id"), str) or not isinstance(block.get("name"), str) or not isinstance(block.get("input", {}), dict):
                                invalid_response(self.name)
                            slot = len(tool_blocks)
                            initial = json.dumps(block["input"]) if block.get("input") else ""
                            argument_bytes += len(initial.encode())
                            if argument_bytes > 1048576:
                                invalid_response(self.name)
                            tool_blocks[index] = {"index": slot, "arguments": initial, "closed": False}
                            yield encoder.encode({"tool_calls": [{"index": slot, "id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": initial}}]})
                        elif block.get("type") != "text":
                            raise GatewayError(502, "unsupported_response", "Unsupported provider content block.", self.name)
                    if ev.get("type") == "content_block_stop" and ev.get("index") in tool_blocks:
                        block = tool_blocks[ev["index"]]
                        if block["closed"]:
                            invalid_response(self.name)
                        arguments(block["arguments"] or "{}", self.name, response=True)
                        if not block["arguments"]:
                            yield encoder.encode({"tool_calls": [{"index": block["index"], "function": {"arguments": "{}"}}]})
                        block["closed"] = True
                    if ev.get("type") == "content_block_delta":
                        delta = ev.get("delta", {})
                        if not isinstance(delta, dict):
                            raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", self.name)
                        if delta.get("type") == "input_json_delta":
                            block = tool_blocks.get(ev.get("index"))
                            partial = delta.get("partial_json")
                            if block is None or block["closed"] or not isinstance(partial, str):
                                invalid_response(self.name)
                            block["arguments"] += partial
                            argument_bytes += len(partial.encode())
                            if argument_bytes > 1048576:
                                invalid_response(self.name)
                            yield encoder.encode({"tool_calls": [{"index": block["index"], "function": {"arguments": partial}}]})
                            continue
                        if delta.get("type") not in (None, "text_delta"):
                            raise GatewayError(502, "unsupported_response", "Unsupported provider content block.", self.name)
                        text = delta.get("text", "")
                        if not isinstance(text, str):
                            raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", self.name)
                        if text:
                            yield encoder.encode({"content": text})
                raise GatewayError(502, "incomplete_stream", "Provider stream ended before completion.", self.name)


class OpenRouterProvider:
    name = "openrouter"
    base = "https://openrouter.ai/api/v1"

    @property
    def available(self):
        return cfg.has_openrouter

    def headers(self):
        return {
            "Authorization": f"Bearer {cfg.openrouter_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-OpenRouter-Title": "anyrouter",
        }

    async def list_models(self):
        return [item["model"] for item in await self.list_catalog()]

    async def list_catalog(self):
        async with client_for(self.name) as client:
            response = await client.get(f"{self.base}/models", headers=self.headers(), timeout=http_timeout(15))
            response.raise_for_status()
            result = []
            for item in response.json()["data"]:
                architecture = item.get("architecture") or {}
                caps = set()
                if "text" in architecture.get("output_modalities", []) and "text" in architecture.get("input_modalities", []):
                    caps.update(("chat", "stream"))
                if "image" in architecture.get("input_modalities", []):
                    caps.add("vision")
                params = item.get("supported_parameters") or []
                if "tools" in params and "tool_choice" in params:
                    caps.add("tools")
                if "response_format" in params:
                    caps.add("json_object")
                if "structured_outputs" in params:
                    caps.add("json_schema")
                pricing = item.get("pricing") or {}
                free = None
                try:
                    if "prompt" in pricing and "completion" in pricing:
                        prices = [Decimal(str(value)) for value in pricing.values()]
                        if all(price.is_finite() and price >= 0 for price in prices):
                            free = all(price == 0 for price in prices)
                except (InvalidOperation, ValueError):
                    pass
                result.append({"model": item["id"], "capabilities": frozenset(caps), "context": positive_int(item.get("context_length")), "free": free})
            return result

    async def chat(self, model, body):
        p = {k: v for k, v in body.items()
             if k in ("messages","temperature","top_p","max_tokens","stop","tools","tool_choice","response_format")}
        p["model"] = model
        if body.get("_automatic"):
            p["provider"] = {"allow_fallbacks": False, "require_parameters": True}
            if not cfg.allow_paid_routing:
                p["provider"]["max_price"] = {"prompt": 0, "completion": 0, "request": 0, "image": 0}
        async with client_for(self.name) as c:
            r = await c.post(f"{self.base}/chat/completions", headers=self.headers(), json=p)
            r.raise_for_status()
            return validate_completion(r.json(), self.name)

    async def stream_chat(self, model, body):
        p = {k: v for k, v in body.items()
             if k in ("messages","temperature","top_p","max_tokens","stop","tools","tool_choice","response_format","stream_options")}
        p["model"] = model
        p["stream"] = True
        if body.get("_automatic"):
            p["provider"] = {"allow_fallbacks": False, "require_parameters": True}
            if not cfg.allow_paid_routing:
                p["provider"]["max_price"] = {"prompt": 0, "completion": 0, "request": 0, "image": 0}
        async with client_for(self.name) as c:
            async with c.stream("POST", f"{self.base}/chat/completions", headers=self.headers(), json=p) as r:
                r.raise_for_status()
                async for chunk in passthrough_sse(r, self.name):
                    yield chunk


class OllamaProvider:
    name = "ollama"

    @property
    def available(self):
        return True

    async def list_models(self):
        async with client_for(self.name) as c:
            r = await c.get(f"{cfg.ollama_url}/api/tags", timeout=http_timeout(3))
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]

    async def list_catalog(self):
        names = await self.list_models()
        from .routing import ollama_is_local
        records = [{"model": name, "free": True if ollama_is_local() else None} for name in names]
        semaphore = asyncio.Semaphore(min(4, cfg.max_concurrency))
        async with client_for(self.name) as client:
            async def details(record):
                async with semaphore:
                    try:
                        response = await client.post(f"{cfg.ollama_url}/api/show", json={"model": record["model"]}, timeout=http_timeout(3))
                        response.raise_for_status()
                        info = response.json()
                        native = info.get("capabilities") or []
                        caps = {"chat", "stream", "json_object", "json_schema"} if "completion" in native else set()
                        if "tools" in native:
                            caps.add("tools")
                        if "vision" in native:
                            caps.add("vision")
                        contexts = [positive_int(value) for key, value in (info.get("model_info") or {}).items() if key.endswith(".context_length")]
                        trained_context = max((c for c in contexts if c), default=None)
                        parameters = info.get("parameters", "")
                        match = re.search(r"(?m)^\s*num_ctx\s+(\d+)\s*$", parameters) if isinstance(parameters, str) else None
                        configured_context = positive_int(int(match[1])) if match else None
                        # The training ceiling is not the serving context. Never assume
                        # a default runtime window or increase GPU allocation implicitly.
                        context = min(configured_context, trained_context) if configured_context and trained_context else configured_context
                        record.update(capabilities=frozenset(caps), context=context)
                    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
                        pass  # Listing remains useful; unknown capabilities cannot AUTO route.
            await asyncio.gather(*(details(record) for record in records[:32]))
        return records

    def _translate(self, model, body, stream):
        return ollama_request(model, body, stream)

    async def chat(self, model, body):
        p = self._translate(model, body, False)
        async with client_for(self.name) as c:
            r = await c.post(f"{cfg.ollama_url}/api/chat", json=p)
            r.raise_for_status()
            resp = r.json()
            if isinstance(resp, dict) and "error" in resp:
                raise GatewayError(502, "provider_error", "Provider request failed.", self.name)
        return ollama_response(model, resp)

    async def stream_chat(self, model, body):
        p = self._translate(model, body, True)
        encoder = ChunkEncoder(model)
        tool_index = 0
        async with client_for(self.name) as c:
            async with c.stream("POST", f"{cfg.ollama_url}/api/chat", json=p) as r:
                r.raise_for_status()
                async for line in bounded_lines(r, self.name):
                    if not line: continue
                    ev = event_json(line, self.name)
                    message = ev.get("message") or {}
                    if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
                        raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", self.name)
                    if message.get("tool_calls"):
                        calls = ollama_calls(message["tool_calls"])
                        for call in calls:
                            yield encoder.encode({"tool_calls": [{"index": tool_index, **call}]})
                            tool_index += 1
                    text = message.get("content", "")
                    if text:
                        yield encoder.encode({"content": text})
                    if ev.get("done") is True:
                        reason = ev.get("done_reason", "stop")
                        if reason not in ("stop", "length"):
                            raise GatewayError(502, "unsupported_response", "Unsupported provider stop reason.", self.name)
                        yield terminal_chunk(encoder, "tool_calls" if tool_index and reason == "stop" else reason, usage_counts(ev.get("prompt_eval_count"), ev.get("eval_count")), body, self.name)
                        yield b"data: [DONE]\n\n"
                        return
                raise GatewayError(502, "incomplete_stream", "Provider stream ended before completion.", self.name)


def positive_int(value):
    return value if type(value) is int and value > 0 else None


def terminal_chunk(encoder, reason, usage, body, provider):
    if body.get("stream_options", {}).get("include_usage") is False:
        if usage is not None:
            from .observability import metrics
            metrics.tokens[provider] += usage["total_tokens"]
        usage = None
    return encoder.encode({}, reason, usage)


PROVIDERS: dict[str, Provider] = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "openrouter": OpenRouterProvider(),
    "ollama": OllamaProvider(),
}


def all_providers():
    return {k: v for k, v in PROVIDERS.items() if v.available}


def get(name):
    return PROVIDERS.get(name)
