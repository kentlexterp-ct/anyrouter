from pathlib import Path

ROOT = Path.home() / "Documents" / "anyrouter"

FILES = {
    "anyrouter/providers.py": '''import json
import httpx
from .config import cfg


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
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(f"{self.base}/models", headers=self.headers())
            r.raise_for_status()
            return [m["id"] for m in r.json()["data"]]

    async def chat(self, model, body):
        p = {k: v for k, v in body.items()
             if k in ("messages", "temperature", "top_p", "max_tokens",
                      "stop", "tools", "tool_choice")}
        p["model"] = model
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{self.base}/chat/completions",
                             headers=self.headers(), json=p)
            r.raise_for_status()
            return r.json()

    async def stream_chat(self, model, body):
        p = {k: v for k, v in body.items()
             if k in ("messages", "temperature", "top_p", "max_tokens",
                      "stop", "tools", "tool_choice")}
        p["model"] = model
        p["stream"] = True
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", f"{self.base}/chat/completions",
                                headers=self.headers(), json=p) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes():
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
        return ["claude-opus-4", "claude-sonnet-4", "claude-haiku-3-5"]

    def _translate(self, body):
        msgs = body.get("messages", [])
        sys_parts, out = [], []
        for m in msgs:
            if m["role"] == "system":
                sys_parts.append(m["content"] if isinstance(m["content"], str)
                                 else json.dumps(m["content"]))
            else:
                out.append({"role": m["role"], "content": m["content"]})
        p = {"model": body["model"], "messages": out,
             "max_tokens": body.get("max_tokens") or 4096}
        if sys_parts:
            p["system"] = "\\n".join(sys_parts)
        if body.get("temperature") is not None:
            p["temperature"] = body["temperature"]
        return p

    def _wrap(self, resp):
        text = "".join(b.get("text", "") for b in resp.get("content", [])
                       if b.get("type") == "text")
        return {
            "id": resp.get("id", "x"),
            "object": "chat.completion",
            "created": 0,
            "model": resp.get("model"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": resp.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": resp.get("usage", {}).get("output_tokens", 0),
                "total_tokens": 0,
            },
        }

    async def chat(self, model, body):
        p = self._translate(body)
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{self.base}/messages",
                             headers=self.headers(), json=p)
            r.raise_for_status()
            return self._wrap(r.json())

    async def stream_chat(self, model, body):
        p = self._translate(body)
        p["stream"] = True
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", f"{self.base}/messages",
                                headers=self.headers(), json=p) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        ev = json.loads(data)
                    except Exception:
                        continue
                    if ev.get("type") == "content_block_delta":
                        text = ev.get("delta", {}).get("text", "")
                        if text:
                            payload = {"choices": [{"delta": {"content": text}, "index": 0}]}
                            yield f"data: {json.dumps(payload)}\\n\\n".encode()
                yield b"data: [DONE]\\n\\n"


class OllamaProvider:
    name = "ollama"

    @property
    def available(self):
        return True

    async def list_models(self):
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{cfg.ollama_url}/api/tags")
                r.raise_for_status()
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    async def chat(self, model, body):
        p = {"model": model, "messages": body.get("messages", []), "stream": False}
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.post(f"{cfg.ollama_url}/api/chat", json=p)
            r.raise_for_status()
            resp = r.json()
        return {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": resp.get("message", {}).get("content", "")},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    async def stream_chat(self, model, body):
        p = {"model": model, "messages": body.get("messages", []), "stream": True}
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", f"{cfg.ollama_url}/api/chat", json=p) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    text = ev.get("message", {}).get("content", "")
                    payload = {"choices": [{"delta": {"content": text}, "index": 0}]}
                    yield f"data: {json.dumps(payload)}\\n\\n".encode()
                    if ev.get("done"):
                        yield b"data: [DONE]\\n\\n"
                        break


PROVIDERS = {
    "openai": OpenAIProvider(),
    "anthropic": AnthropicProvider(),
    "ollama": OllamaProvider(),
}


def all_providers():
    return {k: v for k, v in PROVIDERS.items() if v.available}


def get(name):
    return PROVIDERS.get(name)
''',

    "anyrouter/main.py": '''from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from .config import cfg
from . import providers as prov_mod


_MODEL_INDEX = {}


@asynccontextmanager
async def lifespan(app):
    for name, p in prov_mod.all_providers().items():
        try:
            for m in await p.list_models():
                _MODEL_INDEX.setdefault(m, name)
        except Exception:
            pass
    yield


app = FastAPI(title="anyrouter", lifespan=lifespan)


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_tokens: int | None = None
    stream: bool = False
    stop: list | str | None = None
    tools: list | None = None
    tool_choice: object = None


def resolve(model):
    if "/" in model:
        p, m = model.split("/", 1)
        if prov_mod.get(p):
            return p, m
    if model in _MODEL_INDEX:
        return _MODEL_INDEX[model], model
    for name, p in prov_mod.all_providers().items():
        return name, model
    raise HTTPException(502, "no provider")


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "owned_by": "anyrouter", "provider": p}
            for m, p in _MODEL_INDEX.items()
        ],
    }


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    body = req.model_dump(exclude_none=True)
    prov_name, actual = resolve(body["model"])
    p = prov_mod.get(prov_name)
    if not p or not p.available:
        raise HTTPException(502, f"provider {prov_name} unavailable")

    if req.stream:
        async def gen():
            async for c in p.stream_chat(actual, body):
                yield c
        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        resp = await p.chat(actual, body)
    except Exception as e:
        raise HTTPException(502, f"{prov_name}: {e}")
    return JSONResponse(resp)


@app.get("/")
async def root():
    return {
        "anyrouter": "up",
        "models": len(_MODEL_INDEX),
        "providers": list(prov_mod.all_providers().keys()),
    }


def run():
    import uvicorn
    uvicorn.run("anyrouter.main:app", host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    run()
''',
}


def main():
    for rel, content in FILES.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"[+] {rel}")
    print(f"\nDone. {len(FILES)} files created.")


if __name__ == "__main__":
    main()