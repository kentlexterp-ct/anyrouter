import asyncio
import json
import time
from uuid import uuid4

import anyio
from starlette.responses import JSONResponse, StreamingResponse

from .errors import GatewayError
from .observability import trace


async def until_disconnect(awaitable, request):
    if request is None:
        return await awaitable
    async def invoke():
        return await awaitable

    work = asyncio.create_task(invoke())

    async def disconnected():
        while True:
            if (await request.receive())["type"] == "http.disconnect":
                return

    listener = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((work, listener), return_when=asyncio.FIRST_COMPLETED)
        if listener in done:
            listener.result()
            raise GatewayError(499, "client_disconnected", "Client disconnected.")
        return work.result()
    finally:
        work.cancel()
        listener.cancel()
        with anyio.CancelScope(shield=True):
            await asyncio.gather(work, listener, return_exceptions=True)


class ManagedStreamingResponse(StreamingResponse):
    def __init__(self, content, upstream, deadline, provider=None):
        super().__init__(content, media_type="text/event-stream", headers={
            "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
        })
        self.upstream = upstream
        self.deadline = deadline
        self.provider = provider
        self.started = False
        self.completed = False

    async def stream_response(self, send):
        async def track(message):
            await send(message)
            if message["type"] == "http.response.start":
                self.started = True
            if message["type"] == "http.response.body" and not message.get("more_body", False):
                self.completed = True
        await super().stream_response(track)

    async def __call__(self, scope, receive, send):
        try:
            if self.deadline <= asyncio.get_running_loop().time():
                error = GatewayError(504, "provider_timeout", "Request deadline exceeded.", self.provider)
                if trace.get():
                    trace.get().error = error.code
                await JSONResponse(error.payload(), status_code=504)(scope, receive, send)
                return
            async with anyio.create_task_group() as group:
                async def produce():
                    try:
                        remaining = max(0, self.deadline - asyncio.get_running_loop().time())
                        with anyio.move_on_after(remaining) as deadline_scope:
                            await self.stream_response(send)
                        if deadline_scope.cancel_called and self.started and not self.completed:
                            error = GatewayError(504, "provider_timeout", "Request deadline exceeded.", self.provider)
                            if trace.get():
                                trace.get().error = error.code
                            # Brief bounded grace to terminate a writable response. A
                            # blocked/disconnected client must not hold upstream resources.
                            with anyio.move_on_after(0.1):
                                await send({"type": "http.response.body", "body": f"data: {json.dumps(error.payload())}\n\n".encode(), "more_body": True})
                                await send({"type": "http.response.body", "body": b"", "more_body": False})
                    except OSError:
                        pass
                    finally:
                        group.cancel_scope.cancel()

                async def listen():
                    try:
                        await self.listen_for_disconnect(receive)
                    finally:
                        group.cancel_scope.cancel()

                group.start_soon(produce)
                group.start_soon(listen)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.body_iterator.aclose()
                finally:
                    # Also covers disconnect before the body iterator starts.
                    await self.upstream.aclose()
        if self.background is not None:
            await self.background()


async def bounded_lines(response, provider):
    """Decode fragmented UTF-8 with HTTPX and cap incomplete line buffers."""
    buffer = ""
    async for text in response.aiter_text():
        buffer += text
        while True:
            endings = [pos for pos in (buffer.find("\r"), buffer.find("\n")) if pos >= 0]
            if not endings:
                break
            pos = min(endings)
            if buffer[pos] == "\r" and pos == len(buffer) - 1:
                break  # A fragmented CRLF might continue in the next chunk.
            line = buffer[:pos]
            if len(line.encode()) > 1024 * 1024:
                raise GatewayError(502, "invalid_provider_response", "Provider line is too large.", provider)
            width = 2 if buffer[pos:pos + 2] == "\r\n" else 1
            buffer = buffer[pos + width:]
            yield line
        if len(buffer.encode()) > 1024 * 1024:
            raise GatewayError(502, "invalid_provider_response", "Provider line is too large.", provider)
    if buffer.endswith("\r"):
        yield buffer[:-1]
    elif buffer:
        yield buffer


async def sse_events(response, provider):
    """Parse complete SSE frames, including multiline data and CR/LF endings."""
    lines, data = [], []
    size = 0
    async for line in bounded_lines(response, provider):
        if not line:
            if data:
                yield "\n".join(data), ("\n".join(lines) + "\n\n").encode()
            lines, data, size = [], [], 0
            continue
        size += len(line.encode())
        if size > 1024 * 1024:
            raise GatewayError(502, "invalid_provider_response", "Provider event is too large.", provider)
        lines.append(line)
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)


def event_json(data, provider):
    try:
        event = json.loads(data)
    except ValueError:
        raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", provider) from None
    if not isinstance(event, dict):
        raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", provider)
    if "error" in event or event.get("type") == "error":
        raise GatewayError(502, "provider_error", "Provider stream failed.", provider)
    choices = event.get("choices") or []
    if not isinstance(choices, list):
        raise GatewayError(502, "invalid_provider_response", "Invalid provider stream.", provider)
    if any(isinstance(choice, dict) and choice.get("finish_reason") == "error" for choice in choices):
        raise GatewayError(502, "provider_error", "Provider stream failed.", provider)
    return event


async def passthrough_sse(response, provider):
    async for data, frame in sse_events(response, provider):
        if data == "[DONE]":
            yield b"data: [DONE]\n\n"
            return
        event_json(data, provider)
        yield frame
    raise GatewayError(502, "incomplete_stream", "Provider stream ended before completion.", provider)


class ChunkEncoder:
    def __init__(self, model):
        self.id = f"chatcmpl-{uuid4().hex}"
        self.model = model
        self.created = int(time.time())

    def encode(self, delta, finish_reason=None, usage=None):
        payload = {
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n".encode()


def usage_counts(prompt, completion):
    if all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (prompt, completion)):
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    return None
