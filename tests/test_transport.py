import asyncio
import unittest
from unittest.mock import patch

import httpx

from tests.support import MESSAGES, main, providers
from anyrouter import transport
from anyrouter.config import Config, cfg
from anyrouter.errors import GatewayError


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_clients_are_reused_isolated_and_closed(self):
        original = httpx.AsyncClient
        created, timeouts = [], []
        def handler(request):
            timeouts.append(request.extensions["timeout"])
            return httpx.Response(200, json={"message": {"content": "Hello"}, "choices": [{"message": {"content": "Hello"}}]})
        def factory(*args, **kwargs):
            client = original(*args, transport=httpx.MockTransport(handler), **kwargs)
            created.append(client)
            return client
        with patch.object(transport.httpx, "AsyncClient", side_effect=factory), patch.object(cfg, "openai_key", "synthetic-key"):
            async with transport.lifespan() as runtime:
                adapter = providers.OllamaProvider()
                await adapter.chat("test", {"messages": MESSAGES})
                await adapter.chat("test", {"messages": MESSAGES})
                await providers.OpenAIProvider().chat("test", {"messages": MESSAGES})
                self.assertEqual(len(created), 2)
                self.assertIs(runtime.client("ollama"), created[0])
                self.assertFalse(any(client.is_closed for client in created))
            self.assertTrue(all(client.is_closed for client in created))
            self.assertFalse(runtime.clients)
            self.assertTrue(all(value is not None and value > 0 for timeout in timeouts for value in timeout.values()))

    async def test_lifespan_exception_still_closes_clients(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
        with patch.object(transport.httpx, "AsyncClient", return_value=client):
            with self.assertRaises(RuntimeError):
                async with transport.lifespan() as runtime:
                    runtime.client("ollama")
                    raise RuntimeError("test")
        self.assertTrue(client.is_closed)

    async def test_saturated_provider_rejects_without_running_callback(self):
        started, release = asyncio.Event(), asyncio.Event()
        called = False
        async def held():
            started.set()
            await release.wait()
        async def rejected():
            nonlocal called
            called = True
        with patch.multiple(cfg, max_concurrency=1, queue_timeout=0.05):
            async with transport.lifespan() as runtime:
                task = asyncio.create_task(runtime.call("ollama", held))
                try:
                    await asyncio.wait_for(started.wait(), 1)
                    with self.assertRaises(GatewayError) as caught:
                        await runtime.call("ollama", rejected)
                    self.assertEqual(caught.exception.status_code, 503)
                    self.assertEqual(caught.exception.code, "provider_busy")
                    self.assertFalse(called)
                finally:
                    release.set()
                    await task
                await runtime.call("ollama", rejected)
                self.assertTrue(called)

    async def test_provider_limits_are_independent(self):
        with patch.object(cfg, "max_concurrency", 1):
            async with transport.lifespan() as runtime:
                async with runtime.slot("ollama"):
                    async with runtime.slot("openai"):
                        pass

    async def test_cancellation_and_exceptions_release_admission(self):
        started = asyncio.Event()
        async def held():
            started.set()
            await asyncio.Event().wait()
        async def failed():
            raise RuntimeError("test")
        with patch.object(cfg, "max_concurrency", 1):
            async with transport.lifespan() as runtime:
                task = asyncio.create_task(runtime.call("ollama", held))
                await asyncio.wait_for(started.wait(), 1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                with self.assertRaises(RuntimeError):
                    await runtime.call("ollama", failed)
                async with runtime.slot("ollama"):
                    pass

    async def test_cancelled_queue_wait_does_not_steal_capacity(self):
        with patch.object(cfg, "max_concurrency", 1):
            async with transport.lifespan() as runtime:
                async with runtime.slot("ollama"):
                    waiter = asyncio.create_task(runtime.call("ollama", lambda: asyncio.sleep(0)))
                    await asyncio.sleep(0)
                    waiter.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await waiter
                async with runtime.slot("ollama"):
                    pass

    async def test_overall_nonstream_deadline_cancels_callback(self):
        closed = asyncio.Event()
        async def held():
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
        with patch.object(cfg, "request_timeout", 0.05):
            async with transport.lifespan() as runtime:
                with self.assertRaises(GatewayError) as caught:
                    await runtime.call("ollama", held)
                self.assertEqual(caught.exception.status_code, 504)
                self.assertTrue(closed.is_set())
                async with runtime.slot("ollama"):
                    pass

    async def test_stream_holds_slot_until_closed(self):
        async def source():
            yield b"first"
            yield b"second"
        with patch.multiple(cfg, max_concurrency=1, queue_timeout=0.05):
            async with transport.lifespan() as runtime:
                stream = runtime.stream("ollama", source(), asyncio.get_running_loop().time() + 1)
                self.assertEqual(await anext(stream), b"first")
                with self.assertRaises(GatewayError) as caught:
                    await runtime.call("ollama", lambda: asyncio.sleep(0))
                self.assertEqual(caught.exception.code, "provider_busy")
                await stream.aclose()
                await runtime.call("ollama", lambda: asyncio.sleep(0))

    async def test_stream_overall_deadline_includes_queue_wait(self):
        async def source():
            yield b"first"
        with patch.multiple(cfg, max_concurrency=1, queue_timeout=1):
            async with transport.lifespan() as runtime:
                async with runtime.slot("ollama"):
                    stream = runtime.stream("ollama", source(), asyncio.get_running_loop().time() + 0.05)
                    with self.assertRaises(GatewayError) as caught:
                        await asyncio.wait_for(anext(stream), 0.5)
                    self.assertEqual(caught.exception.code, "provider_timeout")

    async def test_connect_read_write_and_pool_timeouts_are_finite(self):
        timeout = transport.http_timeout()
        self.assertEqual(timeout.connect, cfg.connect_timeout)
        self.assertEqual(timeout.read, cfg.idle_timeout)
        self.assertEqual(timeout.write, cfg.connect_timeout)
        self.assertEqual(timeout.pool, cfg.queue_timeout)

    def test_invalid_reliability_settings_fail_validation(self):
        for field in ("connect_timeout", "first_event_timeout", "idle_timeout", "request_timeout", "queue_timeout"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    Config(**{field: value})
        with self.assertRaises(ValueError):
            Config(max_concurrency=0)
