from contextlib import asynccontextmanager
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
        "data": [{"id": m, "object": "model", "owned_by": "anyrouter", "provider": p}
                 for m, p in _MODEL_INDEX.items()],
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
    return {"anyrouter": "up", "models": len(_MODEL_INDEX),
            "providers": list(prov_mod.all_providers().keys())}


def run():
    import uvicorn
    uvicorn.run("anyrouter.main:app", host=cfg.host, port=cfg.port)


if __name__ == "__main__":
    run()
