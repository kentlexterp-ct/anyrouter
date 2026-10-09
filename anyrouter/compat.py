"""Verified translations for the supported Chat Completions subset."""
import base64
import binascii
import json
import time
from typing import AsyncIterator, Protocol, TypedDict
from uuid import uuid4

from .errors import GatewayError
from .streaming import usage_counts


class ChatBody(TypedDict, total=False):
    model: str
    messages: list[dict]
    temperature: float
    top_p: float
    max_tokens: int
    stream: bool
    stop: str | list[str]
    tools: list[dict]
    tool_choice: str | dict
    response_format: dict
    stream_options: dict


class Provider(Protocol):
    name: str
    @property
    def available(self) -> bool: ...
    async def list_models(self) -> list[str]: ...
    async def chat(self, model: str, body: ChatBody) -> dict: ...
    def stream_chat(self, model: str, body: ChatBody) -> AsyncIterator[bytes]: ...


def unsupported(provider):
    raise GatewayError(422, "unsupported_parameter", "Parameter unsupported by this adapter.", provider)


def invalid_response(provider):
    raise GatewayError(502, "invalid_provider_response", "Invalid provider response.", provider)


def validate_completion(response, provider):
    if not isinstance(response, dict):
        invalid_response(provider)
    if "error" in response:
        raise GatewayError(502, "provider_error", "Provider request failed.", provider)
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        invalid_response(provider)
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            invalid_response(provider)
        if choice.get("finish_reason") == "error":
            raise GatewayError(502, "provider_error", "Provider request failed.", provider)
    return response


def arguments(value, provider, response=False):
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
        if not isinstance(parsed, dict):
            raise ValueError()
        return parsed
    except (ValueError, TypeError):
        if response:
            invalid_response(provider)
        unsupported(provider)


def tools(body, provider):
    result = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            unsupported(provider)
        function = tool["function"]
        if not isinstance(function.get("name"), str) or not function["name"] or not isinstance(function.get("parameters", {}), dict):
            unsupported(provider)
        if set(function) - {"name", "description", "parameters", "strict"} or function.get("strict"):
            unsupported(provider)
        if provider == "anthropic":
            value = {"name": function["name"], "input_schema": function.get("parameters", {"type": "object", "properties": {}})}
            if "description" in function:
                value["description"] = function["description"]
        else:
            value = {"type": "function", "function": {k: v for k, v in function.items() if k != "strict"}}
        result.append(value)
    return result


def content_blocks(content, provider):
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        unsupported(provider)
    result = []
    for block in content:
        if not isinstance(block, dict):
            unsupported(provider)
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            result.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image_url":
            image = block.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str) or not url.startswith("data:image/"):
                # Do not fetch caller-controlled URLs (SSRF and privacy boundary).
                unsupported(provider)
            header, separator, encoded = url.partition(",")
            mime = header[5:].removesuffix(";base64")
            if not separator or not header.endswith(";base64") or mime not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
                unsupported(provider)
            try:
                if not base64.b64decode(encoded, validate=True):
                    unsupported(provider)
            except (ValueError, binascii.Error):
                unsupported(provider)
            result.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": encoded}})
        else:
            unsupported(provider)
    return result


def tool_calls(calls, provider):
    if not isinstance(calls, list):
        unsupported(provider)
    result = []
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function" or not isinstance(call.get("function"), dict) or not isinstance(call.get("id"), str):
            unsupported(provider)
        fn = call["function"]
        if not isinstance(fn.get("name"), str) or not fn["name"]:
            unsupported(provider)
        result.append({"type": "tool_use", "id": call["id"], "name": fn["name"], "input": arguments(fn.get("arguments"), provider)})
    return result


def anthropic_request(body):
    provider = "anthropic"
    system, messages = [], []
    for message in body.get("messages", []):
        role = message["role"]
        if role in ("system", "developer"):
            blocks = content_blocks(message.get("content"), provider)
            if any(b["type"] != "text" for b in blocks):
                unsupported(provider)
            system.extend(b["text"] for b in blocks)
            continue
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                unsupported(provider)
            content = [{"type": "tool_result", "tool_use_id": call_id, "content": content_blocks(message.get("content"), provider)}]
            role = "user"
        elif role in ("user", "assistant"):
            content = content_blocks(message.get("content"), provider)
            if message.get("tool_calls"):
                if role != "assistant":
                    unsupported(provider)
                content.extend(tool_calls(message["tool_calls"], provider))
            if isinstance(message.get("content"), str) and not message.get("tool_calls"):
                content = message["content"]
        else:
            unsupported(provider)
        messages.append({"role": role, "content": content})
    payload = {"model": body["model"], "messages": messages, "max_tokens": body.get("max_tokens") or 4096}
    if system:
        payload["system"] = "\n".join(system)
    temperature, top_p = body.get("temperature"), body.get("top_p")
    if temperature is not None:
        if temperature > 1:
            unsupported(provider)
        payload["temperature"] = temperature
    if top_p not in (None, 1.0):
        if temperature is not None:
            unsupported(provider)
        payload["top_p"] = top_p
    if body.get("stop") is not None:
        payload["stop_sequences"] = [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
    definitions = tools(body, provider)
    if definitions:
        payload["tools"] = definitions
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        if choice.get("type") != "function" or not isinstance(name, str) or name not in {t["name"] for t in definitions}:
            unsupported(provider)
        payload["tool_choice"] = {"type": "tool", "name": name}
    elif choice is not None:
        if choice not in ("auto", "none", "required") or (choice != "none" and not definitions):
            unsupported(provider)
        payload["tool_choice"] = {"type": {"auto": "auto", "none": "none", "required": "any"}[choice]}
    format_ = body.get("response_format", {})
    if format_.get("type") == "json_schema":
        payload["output_config"] = {"format": {"type": "json_schema", "schema": format_["json_schema"]["schema"]}}
    elif format_.get("type") not in (None, "text"):
        unsupported(provider)
    return payload


def ollama_request(model, body, stream):
    provider, messages, names = "ollama", [], {}
    for message in body.get("messages", []):
        role = message["role"]
        if role == "developer":
            role = "system"
        blocks = content_blocks(message.get("content"), provider)
        value = {"role": role, "content": "".join(b["text"] for b in blocks if b["type"] == "text")}
        images = [b["source"]["data"] for b in blocks if b["type"] == "image"]
        if images:
            value["images"] = images
        if message.get("tool_calls"):
            calls = tool_calls(message["tool_calls"], provider)
            value["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["input"]}} for c in calls]
            names.update({c["id"]: c["name"] for c in calls})
        if role == "tool":
            name = names.get(message.get("tool_call_id"))
            if not name:
                unsupported(provider)
            value["tool_name"] = name
        messages.append(value)
    payload = {"model": model, "messages": messages, "stream": stream}
    options = {k: body[k] for k in ("temperature", "top_p") if body.get(k) is not None}
    if body.get("max_tokens") is not None:
        options["num_predict"] = body["max_tokens"]
    if body.get("stop") is not None:
        options["stop"] = [body["stop"]] if isinstance(body["stop"], str) else body["stop"]
    if options:
        payload["options"] = options
    definitions = tools(body, provider)
    choice = body.get("tool_choice")
    if choice not in (None, "auto", "none") or (choice == "auto" and not definitions):
        unsupported(provider)
    if definitions and choice != "none":
        payload["tools"] = definitions
    format_ = body.get("response_format", {})
    if format_.get("type") == "json_object":
        payload["format"] = "json"
    elif format_.get("type") == "json_schema":
        payload["format"] = format_["json_schema"]["schema"]
    return payload


def ollama_calls(calls):
    if not isinstance(calls, list):
        invalid_response("ollama")
    result = []
    for call in calls:
        fn = call.get("function") if isinstance(call, dict) else None
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            invalid_response("ollama")
        result.append({"id": "call_" + uuid4().hex, "type": "function", "function": {"name": fn["name"], "arguments": json.dumps(arguments(fn.get("arguments"), "ollama", response=True), ensure_ascii=False)}})
    return result


def completion(provider, model, message, reason, usage=None, id_=None):
    result = {"id": id_ or "chatcmpl-" + uuid4().hex, "object": "chat.completion", "created": int(time.time()), "model": model,
              "choices": [{"index": 0, "message": message, "finish_reason": reason}]}
    if usage is not None:
        result["usage"] = usage
    return result


def anthropic_response(resp):
    if not isinstance(resp, dict) or not isinstance(resp.get("content"), list):
        invalid_response("anthropic")
    text, calls = [], []
    for block in resp["content"]:
        if not isinstance(block, dict):
            invalid_response("anthropic")
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text.append(block["text"])
        elif block.get("type") == "tool_use" and isinstance(block.get("id"), str) and isinstance(block.get("name"), str):
            calls.append({"id": block["id"], "type": "function", "function": {"name": block["name"], "arguments": json.dumps(arguments(block.get("input"), "anthropic", response=True), ensure_ascii=False)}})
        else:
            raise GatewayError(502, "unsupported_response", "Unsupported provider content block.", "anthropic")
    reason = anthropic_reason(resp.get("stop_reason") or "end_turn")
    message = {"role": "assistant", "content": "".join(text) if text or not calls else None}
    if calls:
        message["tool_calls"] = calls
    usage = resp.get("usage") or {}
    if not isinstance(usage, dict):
        invalid_response("anthropic")
    # Cache read/write input is billed separately but belongs in prompt token totals.
    prompt = usage.get("input_tokens")
    if type(prompt) is int:
        for field in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            extra = usage.get(field, 0)
            if type(extra) is not int or extra < 0:
                invalid_response("anthropic")
            prompt += extra
    return completion("anthropic", resp.get("model"), message, reason, usage_counts(prompt, usage.get("output_tokens")), resp.get("id"))


def anthropic_reason(reason):
    reasons = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}
    if reason not in reasons:
        raise GatewayError(502, "unsupported_response", "Unsupported provider stop reason.", "anthropic")
    return reasons[reason]


def ollama_response(model, resp):
    if not isinstance(resp, dict) or not isinstance(resp.get("message"), dict):
        invalid_response("ollama")
    native = resp["message"]
    text = native.get("content", "")
    if not isinstance(text, str):
        invalid_response("ollama")
    calls = ollama_calls(native["tool_calls"]) if native.get("tool_calls") else []
    message = {"role": "assistant", "content": text if text or not calls else None}
    if calls:
        message["tool_calls"] = calls
    reason = resp.get("done_reason", "stop")
    if reason not in ("stop", "length"):
        raise GatewayError(502, "unsupported_response", "Unsupported provider stop reason.", "ollama")
    return completion("ollama", model, message, "tool_calls" if calls and reason == "stop" else reason,
                      usage_counts(resp.get("prompt_eval_count"), resp.get("eval_count")))
