import asyncio
import unittest
from unittest.mock import patch

import httpx

from tests.support import FakeProvider, MESSAGES, main, providers
from anyrouter import transport
from anyrouter.config import Config, cfg
from anyrouter.errors import GatewayError, provider_error
from anyrouter.routing import Catalog, CircuitBreaker, ModelRecord


class BreakerTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.breaker = CircuitBreaker(lambda: self.now)
        self.settings = patch.multiple(cfg, breaker_threshold=2, breaker_cooldown=10)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    def fail(self, code="provider_error", retry_after=None):
        ticket = self.breaker.acquire("ollama")
        self.breaker.finish(ticket, 1, GatewayError(502, code, "Failure", "ollama", retry_after))

    def test_open_cooldown_single_probe_and_recovery(self):
        self.fail()
        self.assertEqual(self.breaker.state, "closed")
        self.fail()
        with self.assertRaises(GatewayError):
            self.breaker.acquire("ollama")
        self.now = 11
        ticket = self.breaker.acquire("ollama")
        self.assertEqual(self.breaker.state, "half_open")
        with self.assertRaises(GatewayError):
            self.breaker.acquire("ollama")
        self.breaker.finish(ticket, 0.2)
        self.assertEqual(self.breaker.state, "closed")
        self.assertEqual(self.breaker.active, 0)

    def test_late_success_does_not_close_open_circuit(self):
        late = self.breaker.acquire("ollama")
        self.fail()
        self.fail()
        self.breaker.finish(late, 0.1)
        self.assertEqual(self.breaker.state, "open")
        self.assertEqual(self.breaker.active, 0)

    def test_invalid_requests_credentials_busy_and_cancel_are_neutral(self):
        for code in ("provider_auth_error", "provider_rejected_request", "provider_busy", "unsupported_parameter"):
            self.fail(code)
        ticket = self.breaker.acquire("ollama")
        self.breaker.finish(ticket, None)
        self.assertEqual(self.breaker.failures, 0)
        self.assertEqual(self.breaker.state, "closed")

    def test_rate_limit_honors_cooldown_and_failed_probe_reopens(self):
        self.fail("provider_rate_limit", 45)
        self.now = 44
        self.assertFalse(self.breaker.usable())
        self.now = 46
        self.fail()
        self.assertFalse(self.breaker.usable())
        self.assertEqual(self.breaker.state, "open")

    def test_error_categories_never_echo_upstream_body(self):
        expected = {400: "provider_rejected_request", 401: "provider_auth_error", 403: "provider_auth_error", 404: "provider_rejected_request", 429: "provider_rate_limit", 500: "provider_error"}
        for status, code in expected.items():
            response = httpx.Response(status, text="synthetic-secret", request=httpx.Request("POST", "https://example.test"), headers={"Retry-After": "90"})
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                error = provider_error("openai", exc)
            self.assertEqual(error.code, code)
            self.assertNotIn("synthetic-secret", str(error.payload()))
        self.assertEqual(provider_error("ollama", httpx.ConnectError("secret")).code, "provider_connect_error")


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.runtime = transport.Runtime()
        self.catalog = self.runtime.catalog
        self.settings = patch.multiple(cfg, allow_paid_routing=False, route_aliases={}, model_overrides={})
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.body = {"messages": MESSAGES, "max_tokens": 10}

    def add(self, provider="ollama", model="known", **overrides):
        item = {"model": model, "capabilities": frozenset(("chat", "stream")), "context": 10000, "free": True, **overrides}
        self.catalog.replace(provider, [item])

    def test_unknown_capabilities_context_or_cost_are_excluded(self):
        for field, value in (("capabilities", frozenset()), ("context", None), ("free", None), ("free", False)):
            self.add(**{field: value})
            with self.assertRaises(GatewayError) as caught:
                self.catalog.candidates("auto", self.body, self.runtime)
            self.assertEqual(caught.exception.code, "no_eligible_model")

    def test_context_tools_vision_and_structured_requirements(self):
        self.add()
        for body in ({**self.body, "max_tokens": 20000}, {**self.body, "tools": [{"type": "function"}]}, {**self.body, "response_format": {"type": "json_schema"}}, {**self.body, "messages": [{"role": "user", "content": [{"type": "image_url"}]}]}):
            with self.assertRaises(GatewayError):
                self.catalog.candidates("auto", body, self.runtime)

    def test_privacy_and_allowlists_are_enforced(self):
        self.add("openrouter")
        with self.assertRaises(GatewayError):
            self.catalog.candidates("auto", {**self.body, "routing": {"local_only": True}}, self.runtime)
        with self.assertRaises(GatewayError):
            self.catalog.candidates("auto", self.body, self.runtime, lambda p, m: False)
        self.add()
        with patch.object(cfg, "ollama_url", "http://10.0.0.10:11434"):
            with self.assertRaises(GatewayError):
                self.catalog.candidates("auto", {**self.body, "routing": {"local_only": True}}, self.runtime)

    def test_aliases_health_latency_and_staleness_filter_deterministically(self):
        self.add("ollama", "local")
        self.add("openrouter", "remote")
        self.runtime.breaker("ollama").latency = 3
        self.runtime.breaker("openrouter").latency = 1
        self.assertEqual(self.catalog.candidates("auto", self.body, self.runtime)[0].provider, "openrouter")
        with patch.object(cfg, "route_aliases", {"local": ["ollama:local"]}):
            self.assertEqual(len(self.catalog.candidates("local", self.body, self.runtime)), 1)
        self.catalog.records["ollama:local"] = ModelRecord("ollama", "local", frozenset({"chat"}), 10000, True, 0)
        self.runtime.breaker("openrouter").state = "open"
        self.runtime.breaker("openrouter").until = float("inf")
        with self.assertRaises(GatewayError):
            self.catalog.candidates("auto", self.body, self.runtime)

    def test_replace_removes_retired_models_and_failure_keeps_last_good(self):
        self.add(model="old")
        self.add(model="new")
        self.assertNotIn("ollama:old", self.catalog.records)
        self.catalog.failed("ollama", "provider_error")
        self.assertIn("ollama:new", self.catalog.records)
        self.assertFalse(self.catalog.discovery["ollama"]["ok"])

    def test_deployment_metadata_overrides_enable_verified_models(self):
        with patch.object(cfg, "model_overrides", {"ollama:known": {"capabilities": ["chat"], "context": 8192, "free": True}}):
            self.catalog.replace("ollama", ["known"])
        self.assertEqual(self.catalog.candidates("auto", self.body, self.runtime)[0].target, "ollama:known")

    def test_json_modes_and_native_adapter_restrictions_are_distinct(self):
        self.add(capabilities=frozenset({"chat", "tools", "json_schema"}))
        for body in ({**self.body, "response_format": {"type": "json_object"}}, {**self.body, "tools": [{"type": "function", "function": {"name": "x"}}], "tool_choice": "required"}):
            with self.assertRaises(GatewayError):
                self.catalog.candidates("auto", body, self.runtime)
        self.assertEqual(len(self.catalog.candidates("auto", {**self.body, "response_format": {"type": "json_schema"}}, self.runtime)), 1)

    def test_invalid_aliases_and_capability_overrides_fail_configuration(self):
        for configuration in ({"route_aliases": {"ollama/model": ["ollama:model"]}}, {"route_aliases": {"alias": ["ollama:"]}}, {"model_overrides": {"ollama:model": []}}, {"model_overrides": {"ollama:model": {"capabilities": ["invented"]}}}, {"model_overrides": {"ollama:model": {"free": "true"}}}):
            with self.assertRaises(ValueError):
                Config(**configuration)


class RoutingIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        main.keys.windows.clear()
        self.first, self.second = FakeProvider(), FakeProvider()
        self.registry = patch.dict(providers.PROVIDERS, {"ollama": self.first, "openrouter": self.second}, clear=True)
        self.registry.start()
        self.addCleanup(self.registry.stop)
        self.settings = patch.multiple(cfg, route_aliases={}, model_overrides={}, allow_paid_routing=False)
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.context = transport.lifespan()
        self.runtime = await self.context.__aenter__()
        for name in providers.PROVIDERS:
            self.runtime.catalog.replace(name, [{"model": "known", "capabilities": frozenset({"chat", "stream"}), "context": 8192, "free": True}])
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.context.__aexit__(None, None, None)

    async def chat(self, **values):
        return await self.client.post("/v1/chat/completions", json={"model": "auto", "messages": MESSAGES, "max_tokens": 10, **values})

    async def test_safe_connection_failure_falls_back_before_output(self):
        for stream in (False, True):
            self.first.error = httpx.ConnectError("synthetic-secret")
            response = await self.chat(stream=stream)
            self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.second.calls), 2)

    async def test_auto_output_budget_matches_context_admission(self):
        response = await self.client.post("/v1/chat/completions", json={"model": "auto", "messages": MESSAGES})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.first.calls[0][1]["max_tokens"], 4096)

    async def test_explicit_route_never_switches_and_unknown_is_404(self):
        self.first.error = httpx.ConnectError("secret")
        self.assertEqual((await self.chat(model="ollama:known")).status_code, 502)
        self.assertFalse(self.second.calls)
        with patch.dict(main._MODEL_INDEX, {}, clear=True):
            self.assertEqual((await self.chat(model="unknown")).status_code, 404)

    async def test_ambiguous_timeout_never_replays_generation(self):
        self.first.error = httpx.ReadTimeout("secret")
        response = await self.chat()
        self.assertEqual(response.status_code, 504)
        self.assertFalse(self.second.calls)

    async def test_partial_stream_failure_does_not_switch(self):
        self.first.fail_after = 1
        response = await self.chat(stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("provider_error", response.text)
        self.assertNotIn("[DONE]", response.text)
        self.assertFalse(self.second.calls)

    async def test_circuit_blocks_callback_and_recovers_with_one_probe(self):
        self.first.error = RuntimeError("secret")
        with patch.multiple(cfg, breaker_threshold=1, breaker_cooldown=0.01):
            await self.chat(model="ollama:known")
            response = await self.chat(model="ollama:known")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(len(self.first.calls), 1)
            await asyncio.sleep(0.02)
            self.first.error = None
            self.assertEqual((await self.chat(model="ollama:known")).status_code, 200)
            self.assertEqual(self.runtime.breaker("ollama").state, "closed")

    async def test_readiness_distinguishes_failed_discovery(self):
        self.assertEqual((await self.client.get("/ready")).status_code, 200)
        for name in providers.PROVIDERS:
            self.runtime.catalog.failed(name, "provider_error")
        self.assertEqual((await self.client.get("/ready")).status_code, 503)

    async def test_background_refresh_updates_model_index_and_stops_on_shutdown(self):
        refreshed, calls = asyncio.Event(), []
        async def changing():
            calls.append(1)
            if len(calls) > 1:
                refreshed.set()
            return ["old" if len(calls) == 1 else "new"]
        self.first.list_models = changing
        with patch.object(cfg, "catalog_refresh", 0.01):
            async with main.lifespan(main.app):
                self.assertIn("old", main._MODEL_INDEX)
                await asyncio.wait_for(refreshed.wait(), 1)
                await asyncio.sleep(0.01)
                self.assertNotIn("old", main._MODEL_INDEX)
                self.assertIn("new", main._MODEL_INDEX)
            count = len(calls)
            await asyncio.sleep(0.02)
            self.assertEqual(len(calls), count)
