import asyncio
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

import httpx
import uvicorn

from tests.support import FakeProvider, MESSAGES, main, providers
from anyrouter import bootsrap2, transport
from anyrouter.config import cfg
from anyrouter.errors import GatewayError
from anyrouter.security import GatewayMiddleware


class BootstrapTests(unittest.TestCase):
    def test_dry_run_writes_nothing_and_existing_targets_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "new"
            with redirect_stdout(io.StringIO()):
                bootsrap2.main(["--target", str(target)])
            self.assertFalse(target.exists())
            target.mkdir()
            existing = target / "existing.txt"
            existing.write_text("preserve", encoding="utf-8")
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bootsrap2.main(["--target", str(target), "--create"])
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")

    def test_path_traversal_is_refused_before_creating_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(bootsrap2.FILES, {"../outside.py": "invalid"}, clear=True):
            target = Path(directory) / "new"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                bootsrap2.main(["--target", str(target), "--create"])
            self.assertFalse(target.exists())


class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_global_request_bound_covers_slow_requests(self):
        started, release = asyncio.Event(), asyncio.Event()
        async def application(scope, receive, send):
            started.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})
        middleware = GatewayMiddleware(application)
        setting = patch.object(cfg, "max_requests", 1)
        setting.start()
        self.addCleanup(setting.stop)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=middleware), base_url="http://test") as client:
            first = asyncio.create_task(client.get("/"))
            try:
                await asyncio.wait_for(started.wait(), 1)
                second = await client.get("/")
                self.assertEqual(second.status_code, 503)
                self.assertEqual(second.json()["error"]["code"], "gateway_busy")
            finally:
                release.set()
                await first
            self.assertEqual(middleware.active, 0)

    async def test_body_read_deadline_releases_global_admission(self):
        async def application(scope, receive, send):
            self.fail("Timed-out body must not reach the application")
        middleware = GatewayMiddleware(application)
        sent = []
        async def receive():
            await asyncio.Event().wait()
        async def send(message):
            sent.append(message)
        with patch.object(cfg, "request_timeout", 0.01):
            await middleware({"type": "http", "method": "POST", "path": "/", "headers": [], "client": ("127.0.0.1", 1)}, receive, send)
        self.assertEqual(sent[0]["status"], 408)
        self.assertEqual(middleware.active, 0)

    async def test_credential_failure_is_suppressed_until_rotation(self):
        runtime, calls = transport.Runtime(), 0
        async def invalid():
            nonlocal calls
            calls += 1
            response = httpx.Response(401, request=httpx.Request("POST", "https://example.test"))
            response.raise_for_status()
        with patch.object(cfg, "openai_key", "synthetic-old-key"):
            with self.assertRaises(httpx.HTTPStatusError):
                await runtime.call("openai", invalid)
            with self.assertRaises(GatewayError) as caught:
                await runtime.call("openai", invalid)
            self.assertEqual(caught.exception.code, "provider_auth_error")
            self.assertEqual(calls, 1)
            with patch.object(cfg, "openai_key", "synthetic-new-key"):
                self.assertFalse(runtime.credential_blocked("openai"))
        self.assertEqual(runtime.breaker("openai").failures, 0)


class SocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_stream_and_disconnect_release_provider(self):
        class SocketProvider(FakeProvider):
            def __init__(self):
                super().__init__()
                self.released = asyncio.Event()
            async def stream_chat(self, model, body):
                try:
                    yield self.chunks[0]
                    await asyncio.Event().wait()
                finally:
                    self.closed = True
                    self.released.set()
        provider = SocketProvider()
        main.keys.windows.clear()
        with patch.dict(providers.PROVIDERS, {"ollama": provider}, clear=True), patch.multiple(cfg, key_file=None, auth_required=False, host="127.0.0.1", request_timeout=2):
            server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=0, log_level="error", access_log=False, ws="none", timeout_graceful_shutdown=1))
            listener = socket.socket()
            listener.bind(("127.0.0.1", 0))
            listener.listen(8)
            task = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                for _ in range(100):
                    if server.started:
                        break
                    if task.done():
                        task.result()
                    await asyncio.sleep(0.01)
                self.assertTrue(server.started)
                port = listener.getsockname()[1]
                async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=2) as client:
                    self.assertEqual((await client.get("/ready")).status_code, 200)
                    async with client.stream("POST", "/v1/chat/completions", json={"model": "ollama:known", "messages": MESSAGES, "stream": True}) as response:
                        self.assertEqual(response.status_code, 200)
                        self.assertIn("x-request-id", response.headers)
                        first = await anext(response.aiter_lines())
                        self.assertIn("Hello", first)
                    await asyncio.wait_for(provider.released.wait(), 1)
                    self.assertTrue(provider.closed)
                    self.assertEqual(transport.current_runtime().breaker("ollama").active, 0)
            finally:
                server.should_exit = True
                try:
                    await asyncio.wait_for(task, 3)
                finally:
                    listener.close()
