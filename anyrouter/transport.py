import asyncio
import time
import json
import hashlib
from contextlib import asynccontextmanager

import anyio
import httpx

from .config import cfg
from .errors import GatewayError, provider_error
from .routing import CircuitBreaker, Catalog
from .observability import upstream_result, metrics, trace


def http_timeout(read=None):
    return httpx.Timeout(
        connect=cfg.connect_timeout, read=cfg.idle_timeout if read is None else read,
        write=cfg.connect_timeout, pool=cfg.queue_timeout,
    )


class Runtime:
    """One client pool and concurrency budget per provider, per worker."""

    def __init__(self):
        self.clients = {}
        self.semaphores = {}
        self.breakers = {}
        self.catalog = Catalog()
        self.credential_failures = {}

    def breaker(self, provider):
        return self.breakers.setdefault(provider, CircuitBreaker())

    def credential_version(self, provider):
        value = getattr(cfg, provider + "_key", None)
        if provider == "ollama":
            value = cfg.ollama_url
        return hashlib.sha256((value or "").encode()).digest()

    def credential_blocked(self, provider):
        return provider in self.credential_failures and self.credential_failures[provider] == self.credential_version(provider)

    @asynccontextmanager
    async def attempt(self, provider):
        if trace.get():
            trace.get().provider = provider
        if self.credential_blocked(provider):
            raise GatewayError(502, "provider_auth_error", "Provider credentials were rejected.", provider)
        credential_version = self.credential_version(provider)
        breaker = self.breaker(provider)
        ticket = breaker.acquire(provider)
        started, error = time.monotonic(), None
        completed = False
        try:
            yield
            completed = True
        except Exception as exc:
            error = provider_error(provider, exc)
            if error.code == "provider_auth_error":
                self.credential_failures[provider] = credential_version
            raise
        finally:
            elapsed = upstream_result(provider, started, error) if completed or error else None
            breaker.finish(ticket, elapsed, error)

    def client(self, provider):
        if provider not in self.clients:
            self.clients[provider] = httpx.AsyncClient(
                timeout=http_timeout(),
                limits=httpx.Limits(
                    max_connections=cfg.max_concurrency,
                    max_keepalive_connections=cfg.max_concurrency,
                ),
            )
        return self.clients[provider]

    async def close(self):
        with anyio.CancelScope(shield=True):
            results = await asyncio.gather(
                *(client.aclose() for client in self.clients.values()), return_exceptions=True,
            )
        self.clients.clear()
        self.semaphores.clear()
        for result in results:
            if isinstance(result, BaseException):
                raise result

    @asynccontextmanager
    async def slot(self, provider, deadline=None):
        semaphore = self.semaphores.setdefault(provider, asyncio.Semaphore(cfg.max_concurrency))
        queue_timeout = cfg.queue_timeout
        overall = False
        if deadline is not None:
            remaining = deadline - asyncio.get_running_loop().time()
            overall = remaining <= queue_timeout
            queue_timeout = min(queue_timeout, max(0, remaining))
        try:
            await asyncio.wait_for(semaphore.acquire(), queue_timeout)
        except asyncio.TimeoutError:
            if overall:
                raise GatewayError(504, "provider_timeout", "Provider request deadline exceeded.", provider) from None
            raise GatewayError(503, "provider_busy", "Provider concurrency limit reached.", provider) from None
        try:
            yield
        finally:
            semaphore.release()

    async def call(self, provider, callback, deadline=None):
        deadline = deadline or asyncio.get_running_loop().time() + cfg.request_timeout
        async def invoke():
            async with self.slot(provider, deadline):
                return await callback()
        async with self.attempt(provider):
            try:
                result = await asyncio.wait_for(invoke(), max(0, deadline - asyncio.get_running_loop().time()))
                usage = result.get("usage") if isinstance(result, dict) else None
                if isinstance(usage, dict) and type(usage.get("total_tokens")) is int and usage["total_tokens"] >= 0:
                    metrics.tokens[provider] += usage["total_tokens"]
                return result
            except asyncio.TimeoutError:
                raise GatewayError(504, "provider_timeout", "Provider request deadline exceeded.", provider) from None

    async def stream(self, provider, source, deadline):
        observed_tokens = 0
        try:
            async with self.attempt(provider):
                try:
                    async with self.slot(provider, deadline):
                        timeout = cfg.first_event_timeout
                        while True:
                            remaining = deadline - asyncio.get_running_loop().time()
                            if remaining <= 0:
                                raise asyncio.TimeoutError
                            try:
                                chunk = await asyncio.wait_for(anext(source), min(timeout, remaining))
                            except StopAsyncIteration:
                                return
                            # Usage is cumulative; retain the latest valid total once.
                            try:
                                data = "\n".join(line[5:].lstrip(" ") for line in chunk.decode().splitlines() if line.startswith("data:"))
                                record = json.loads(data) if data and data != "[DONE]" else {}
                                usage = record.get("usage") if isinstance(record, dict) else None
                                total = usage.get("total_tokens") if isinstance(usage, dict) else None
                                if type(total) is int and total >= 0:
                                    # Provider adapters emit one terminal usage record; some
                                    # compatible upstreams repeat cumulative usage snapshots.
                                    metrics.tokens[provider] += max(0, total - observed_tokens)
                                    observed_tokens = max(observed_tokens, total)
                            except (ValueError, UnicodeDecodeError):
                                pass  # Framing and content are validated by the adapter.
                            yield chunk
                            timeout = cfg.idle_timeout
                except asyncio.TimeoutError:
                    raise GatewayError(504, "provider_timeout", "Provider stream deadline exceeded.", provider) from None
        finally:
            with anyio.CancelScope(shield=True):
                await source.aclose()


_runtime = None


def current_runtime():
    # Standalone adapter/route calls can run outside ASGI lifespan (e.g. tests).
    return _runtime if _runtime is not None else Runtime()


@asynccontextmanager
async def lifespan():
    global _runtime
    previous, runtime = _runtime, Runtime()
    _runtime = runtime
    try:
        yield runtime
    finally:
        _runtime = previous
        await runtime.close()


@asynccontextmanager
async def client_for(provider):
    if _runtime is not None:
        yield _runtime.client(provider)
    else:
        runtime = Runtime()
        try:
            yield runtime.client(provider)
        finally:
            await runtime.close()
