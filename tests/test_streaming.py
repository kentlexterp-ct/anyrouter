import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from starlette.requests import Request

from tests.support import FakeProvider, MESSAGES, main, providers
from anyrouter import transport
from anyrouter.config import cfg
from anyrouter.errors import GatewayError
from anyrouter.streaming import passthrough_sse


class FragmentedStream(httpx.AsyncByteStream):
    def __init__(self, payload):
        self.payload = payload
        self.closed = False

    async def __aiter__(self):
        for byte in self.payload:
            yield bytes([byte])

    async def aclose(self):
        self.closed = True


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    async def test_fragmented_utf8_multiline_sse_and_comments(self):
        raw = ': keepalive\r\n\r\nevent: message\r\ndata:{"choices":\r\ndata: [{"delta":{"content":"你好"}}]}\r\n\r\ndata:[DONE]\r\n\r\n'
        stream = FragmentedStream(raw.encode())
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))) as client:
            async with client.stream("POST", "https://example.test") as response:
                chunks = [chunk async for chunk in passthrough_sse(response, "test")]
        self.assertEqual(len(chunks), 2)
        self.assertIn("你好", chunks[0].decode())
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")
        self.assertTrue(stream.closed)

    async def test_passthrough_errors_malformed_events_and_missing_done(self):
        for text, expected in (
            ('data: {"error":{"message":"synthetic-secret"}}\n\n', "provider_error"),
            ('data: {"choices":[{"finish_reason":"error"}]}\n\n', "provider_error"),
            ('data: broken\n\n', "invalid_provider_response"),
            ('data: []\n\n', "invalid_provider_response"),
            ('data: {"choices":[]}\n\n', "incomplete_stream"),
            ('data: [DONE]', "incomplete_stream"),
        ):
            with self.subTest(text=text):
                response = httpx.Response(200, text=text)
                chunks = []
                with self.assertRaises(GatewayError) as caught:
                    async for chunk in passthrough_sse(response, "test"):
                        chunks.append(chunk)
                self.assertEqual(caught.exception.code, expected)
                self.assertNotIn("synthetic-secret", str(caught.exception))
                self.assertFalse(any(b"[DONE]" in chunk for chunk in chunks))

    async def test_provider_frames_preserve_finish_usage_and_tool_deltas(self):
        payload = {"id": "test", "choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":"}}]}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        response = httpx.Response(200, text=f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n")
        chunks = [chunk async for chunk in passthrough_sse(response, "test")]
        self.assertEqual(json.loads(chunks[0].decode()[6:]), payload)

    async def test_unterminated_oversized_line_is_rejected(self):
        class OversizedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for _ in range(1025):
                    yield b"x" * 1024
        response = httpx.Response(200, stream=OversizedStream())
        try:
            with self.assertRaises(GatewayError) as caught:
                async for _ in passthrough_sse(response, "test"):
                    pass
            self.assertEqual(caught.exception.code, "invalid_provider_response")
        finally:
            await response.aclose()

    async def test_lone_cr_sse_frames_are_supported(self):
        response = httpx.Response(200, text='data: {"choices":[]}\r\rdata: [DONE]\r\r')
        chunks = [chunk async for chunk in passthrough_sse(response, "test")]
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")

    async def test_anthropic_and_ollama_preserve_stream_metadata(self):
        original = httpx.AsyncClient
        events = [
            {"type": "message_start", "message": {"id": "upstream-id", "model": "test-model", "usage": {"input_tokens": 10, "output_tokens": 0}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 5}},
            {"type": "message_stop"},
        ]
        anthropic = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        ollama = json.dumps({"message": {"content": "Hello"}, "done": True, "done_reason": "length", "prompt_eval_count": 10, "eval_count": 5}) + "\n"
        def handler(request):
            return httpx.Response(200, text=anthropic if request.url.path.endswith("/messages") else ollama)
        def factory(*args, **kwargs):
            return original(*args, transport=httpx.MockTransport(handler), **kwargs)
        with patch.object(transport.httpx, "AsyncClient", side_effect=factory), patch.object(cfg, "anthropic_key", "synthetic-key"):
            for adapter in (providers.AnthropicProvider(), providers.OllamaProvider()):
                with self.subTest(provider=adapter.name):
                    chunks = [chunk async for chunk in adapter.stream_chat("test-model", {"messages": MESSAGES})]
                    records = [json.loads(chunk.decode()[6:]) for chunk in chunks[:-1]]
                    self.assertEqual(len({record["id"] for record in records}), 1)
                    self.assertTrue(all(record["object"] == "chat.completion.chunk" for record in records))
                    self.assertEqual(records[-1]["choices"][0]["finish_reason"], "length")
                    self.assertEqual(records[-1]["usage"]["total_tokens"], 15)
                    self.assertEqual(chunks[-1], b"data: [DONE]\n\n")


class BlockingProvider(FakeProvider):
    def __init__(self, block_first=True):
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block_first = block_first

    async def chat(self, model, body):
        self.started.set()
        try:
            await self.release.wait()
            return self.result
        finally:
            self.closed = True

    async def stream_chat(self, model, body):
        self.started.set()
        try:
            if self.block_first:
                await self.release.wait()
            yield self.chunks[0]
            await self.release.wait()
            yield self.chunks[-1]
        finally:
            self.closed = True


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        main.keys.windows.clear()
        self.provider = BlockingProvider()
        self.registry = patch.dict(providers.PROVIDERS, {"ollama": self.provider}, clear=True)
        self.registry.start()
        self.settings = patch.multiple(cfg, request_timeout=1, first_event_timeout=0.05, idle_timeout=0.05, max_concurrency=1, queue_timeout=0.05)
        self.settings.start()
        self.context = transport.lifespan()
        self.runtime = await self.context.__aenter__()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.context.__aexit__(None, None, None)
        self.settings.stop()
        self.registry.stop()

    async def chat(self, stream=True):
        return await self.client.post("/v1/chat/completions", json={"model": "ollama/test", "messages": MESSAGES, "stream": stream})

    async def test_first_output_timeout_returns_504_and_closes_source(self):
        response = await self.chat()
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error"]["code"], "provider_timeout")
        self.assertTrue(self.provider.closed)
        async with self.runtime.slot("ollama"):
            pass

    async def test_idle_timeout_after_output_is_sse_error_without_done(self):
        self.provider.block_first = False
        response = await self.chat()
        self.assertEqual(response.status_code, 200)
        self.assertIn("Hello", response.text)
        self.assertIn("provider_timeout", response.text)
        self.assertNotIn("[DONE]", response.text)
        self.assertTrue(self.provider.closed)

    async def test_overall_stream_deadline_terminates_writable_response(self):
        self.provider.block_first = False
        with patch.multiple(cfg, request_timeout=0.05, idle_timeout=1):
            response = await self.chat()
        self.assertEqual(response.status_code, 200)
        self.assertIn("provider_timeout", response.text)
        self.assertNotIn("[DONE]", response.text)
        self.assertTrue(self.provider.closed)

    async def test_webhook_overall_timeout_keeps_explicit_fallback(self):
        with patch.object(cfg, "request_timeout", 0.05):
            response = await self.client.post("/webhook/classify", json={"message": "urgent"})
        self.assertEqual(response.status_code, 504)
        self.assertTrue(response.json()["fallback"])
        self.assertTrue(self.provider.closed)

    async def test_saturation_returns_503_for_chat_and_webhook(self):
        async with self.runtime.slot("ollama"):
            response = await self.chat(False)
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["error"]["code"], "provider_busy")
            response = await self.client.post("/webhook/classify", json={})
            self.assertEqual(response.status_code, 503)
            self.assertTrue(response.json()["fallback"])
        self.assertFalse(self.provider.started.is_set())

    async def test_cancel_before_first_output_closes_source_and_releases_slot(self):
        with patch.object(cfg, "first_event_timeout", 1):
            task = asyncio.create_task(main.chat(main.ChatRequest(model="ollama/test", messages=MESSAGES, stream=True)))
            await asyncio.wait_for(self.provider.started.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(self.provider.closed)
        async with self.runtime.slot("ollama"):
            pass

    async def test_disconnect_before_first_output_cancels_upstream(self):
        queue = asyncio.Queue()
        request = Request({"type": "http"}, receive=queue.get)
        with patch.object(cfg, "first_event_timeout", 1):
            task = asyncio.create_task(main.chat(main.ChatRequest(model="ollama/test", messages=MESSAGES, stream=True), request))
            await asyncio.wait_for(self.provider.started.wait(), 1)
            await queue.put({"type": "http.disconnect"})
            with self.assertRaises(GatewayError) as caught:
                await task
        self.assertEqual(caught.exception.code, "client_disconnected")
        self.assertTrue(self.provider.closed)

    async def test_disconnect_after_headers_or_body_closes_prefetched_source(self):
        for after_body in (False, True):
            with self.subTest(after_body=after_body):
                self.provider = BlockingProvider(block_first=False)
                providers.PROVIDERS["ollama"] = self.provider
                response = await main.chat(main.ChatRequest(model="ollama/test", messages=MESSAGES, stream=True))
                queue = asyncio.Queue()
                async def send(message):
                    if message["type"] == ("http.response.body" if after_body else "http.response.start"):
                        await queue.put({"type": "http.disconnect"})
                        await asyncio.Event().wait()
                await asyncio.wait_for(response({"type": "http", "asgi": {"spec_version": "2.4"}}, queue.get, send), 1)
                self.assertTrue(self.provider.closed)
                async with self.runtime.slot("ollama"):
                    pass

    async def test_slow_downstream_cannot_hold_stream_indefinitely(self):
        self.provider.block_first = False
        self.provider.release.set()
        with patch.object(cfg, "request_timeout", 0.05):
            response = await main.chat(main.ChatRequest(model="ollama/test", messages=MESSAGES, stream=True))
            async def receive():
                await asyncio.Event().wait()
            async def send(message):
                if message["type"] == "http.response.body":
                    await asyncio.Event().wait()
            await asyncio.wait_for(response({"type": "http"}, receive, send), 0.5)
        self.assertTrue(self.provider.closed)
        async with self.runtime.slot("ollama"):
            pass

    async def test_expired_deadline_before_headers_returns_504_and_cleans_up(self):
        self.provider.block_first = False
        response = await main.chat(main.ChatRequest(model="ollama/test", messages=MESSAGES, stream=True))
        response.deadline = asyncio.get_running_loop().time() - 1
        sent = []
        async def send(message):
            sent.append(message)
        async def receive():
            await asyncio.Event().wait()
        await response({"type": "http"}, receive, send)
        self.assertEqual(sent[0]["status"], 504)
        self.assertTrue(self.provider.closed)
