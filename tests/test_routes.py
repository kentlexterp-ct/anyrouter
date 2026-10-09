import json
import unittest
from unittest.mock import patch

import httpx

from tests.support import FakeProvider, MESSAGES, main, providers


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        main.keys.windows.clear()
        self.provider = FakeProvider()
        self.registry = patch.dict(providers.PROVIDERS, {"ollama": self.provider}, clear=True)
        self.registry.start()
        self.index = patch.dict(main._MODEL_INDEX, {}, clear=True)
        self.index.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app), base_url="http://test",
        )

    async def asyncTearDown(self):
        await self.client.aclose()
        self.index.stop()
        self.registry.stop()

    async def chat(self, **overrides):
        return await self.client.post(
            "/v1/chat/completions", json={"model": "test-model", "messages": MESSAGES, **overrides},
        )

    async def classify(self, payload=None):
        return await self.client.post("/webhook/classify", json={} if payload is None else payload)

    async def test_health_discovery_and_models_remain_available(self):
        async with main.lifespan(main.app):
            health = (await self.client.get("/")).json()
            models = (await self.client.get("/v1/models")).json()
        self.assertEqual(health, {"anyrouter": "up", "models": 1, "providers": ["ollama"]})
        self.assertEqual(models["data"][0]["id"], "test-model")
        self.assertEqual(models["data"][0]["provider"], "ollama")

    async def test_chat_preserves_success_response_and_explicit_routing(self):
        response = await self.chat(model="ollama/test-model", temperature=0.2, max_tokens=10)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), self.provider.result)
        self.assertEqual(self.provider.calls[0][0], "test-model")
        self.assertEqual(self.provider.calls[0][1]["max_tokens"], 10)

    async def test_ollama_chat_default_sampling_is_preserved(self):
        self.assertEqual((await self.chat()).status_code, 200)
        body = self.provider.calls[0][1]
        self.assertEqual(body["temperature"], 1.0)
        self.assertEqual(body["top_p"], 1.0)

    async def test_chat_model_validation(self):
        for model in ("", " ", "ollama/"):
            with self.subTest(model=model):
                response = await self.chat(model=model)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["error"]["code"], "invalid_request")
        self.assertFalse(self.provider.calls)

    async def test_chat_message_validation(self):
        for messages in ([], [{}], ["text"], [{"content": "Hi"}], [{"role": "user", "content": 12}]):
            with self.subTest(messages=messages):
                response = await self.chat(messages=messages)
                self.assertEqual(response.status_code, 422)
        self.assertFalse(self.provider.calls)

    async def test_chat_rejects_invalid_numeric_parameters(self):
        for values in ({"max_tokens": 0}, {"max_tokens": -1}, {"temperature": -0.1}, {"top_p": 0}, {"top_p": 1.1}):
            with self.subTest(values=values):
                self.assertEqual((await self.chat(**values)).status_code, 422)
        self.assertFalse(self.provider.calls)

    async def test_chat_rejects_unknown_parameters_without_echoing_input(self):
        response = await self.chat(response_format={"synthetic-secret": "value"})
        self.assertEqual(response.status_code, 422)
        self.assertNotIn("synthetic-secret", response.text)
        self.assertFalse(self.provider.calls)

    async def test_openai_style_assistant_tool_calls_are_preserved(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]}]
        response = await self.chat(messages=messages)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.provider.calls[0][1]["messages"], messages)

    async def test_chat_provider_errors_are_sanitized(self):
        self.provider.error = RuntimeError("synthetic-secret")
        response = await self.chat()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "provider_error")
        self.assertNotIn("synthetic-secret", response.text)

    async def test_chat_timeout_returns_504(self):
        self.provider.error = httpx.ReadTimeout("synthetic-secret")
        response = await self.chat()
        self.assertEqual(response.status_code, 504)
        self.assertEqual(response.json()["error"]["code"], "provider_timeout")

    async def test_chat_unavailable_and_missing_providers_are_controlled(self):
        self.provider.available = False
        self.assertEqual((await self.chat(model="ollama/test-model")).json()["error"]["code"], "provider_unavailable")
        providers.PROVIDERS.clear()
        response = await self.chat()
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "no_provider")

    async def test_valid_labels_preserve_webhook_fields(self):
        for raw, expected in (("hot", "hot"), (" WARM\n", "warm"), ("Cold", "cold")):
            with self.subTest(raw=raw):
                self.provider.result = {"choices": [{"message": {"content": raw}}]}
                response = await self.classify({"first_name": "Jane", "last_name": "Ready", "message": "urgent"})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), {"classification": expected, "raw": expected, "name": "Jane Ready", "signals_present": True, "brief": "Name: Jane Ready\nMessage from lead: urgent", "fallback": False})

    async def test_aliases_numeric_budget_and_whitespace_are_preserved(self):
        response = await self.classify({"source": " ", "utm_source": "blog", "company_name": "Acme", "notes": "Interested", "budget": 0})
        brief = response.json()["brief"]
        for text in ("Lead source: blog", "Company: Acme", "Message from lead: Interested", "Budget / timeline: 0"):
            self.assertIn(text, brief)

    async def test_webhook_model_and_generation_controls_are_preserved(self):
        await self.classify({"email": "test@example.test"})
        model, body = self.provider.calls[0]
        self.assertEqual(model, "llama3.2:latest")
        self.assertEqual(body["temperature"], 0.1)
        self.assertEqual(body["max_tokens"], 10)

    async def test_webhook_malformed_json_returns_400_without_calling_provider(self):
        response = await self.client.post("/webhook/classify", content="{", headers={"content-type": "application/json"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["code"], "invalid_json")
        self.assertFalse(self.provider.calls)

    async def test_webhook_non_object_json_returns_422(self):
        for payload in ([], None, "lead", 7):
            with self.subTest(payload=payload):
                response = await self.client.post("/webhook/classify", content=json.dumps(payload), headers={"content-type": "application/json"})
                self.assertEqual(response.status_code, 422)
        self.assertFalse(self.provider.calls)

    async def test_webhook_rejects_nested_or_boolean_lead_fields(self):
        for value in ({"text": "Hi"}, ["Hi"], True):
            with self.subTest(value=value):
                self.assertEqual((await self.classify({"message": value})).status_code, 422)
        self.assertFalse(self.provider.calls)

    async def test_invalid_labels_produce_explicit_fallback(self):
        for raw in ("not hot", "snapshot", "hot or cold", "warm.", "unrelated", ""):
            with self.subTest(raw=raw):
                self.provider.result = {"choices": [{"message": {"content": raw}}]}
                response = await self.classify()
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json()["classification"], "warm")
                self.assertTrue(response.json()["fallback"])
                self.assertEqual(response.json()["error"]["code"], "invalid_classification")

    async def test_malformed_classifier_responses_are_controlled(self):
        for result in ({}, {"choices": []}, {"choices": [{"message": {"content": None}}]}, None):
            with self.subTest(result=result):
                self.provider.result = result
                response = await self.classify()
                self.assertEqual(response.status_code, 502)
                self.assertEqual(response.json()["error"]["code"], "invalid_provider_response")

    async def test_classifier_failure_schema_is_stable_and_sanitized(self):
        self.provider.error = RuntimeError("synthetic-secret")
        response = await self.classify({"first_name": "Jane"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(set(response.json()), {"classification", "raw", "name", "signals_present", "brief", "fallback", "error"})
        self.assertTrue(response.json()["fallback"])
        self.assertNotIn("synthetic-secret", response.text)

    async def test_classifier_timeout_and_unavailability_are_distinct(self):
        self.provider.error = httpx.ReadTimeout("synthetic-secret")
        self.assertEqual((await self.classify()).status_code, 504)
        self.provider.available = False
        response = await self.classify()
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json()["fallback"])
        providers.PROVIDERS.clear()
        self.assertEqual((await self.classify()).status_code, 503)

    async def test_successful_stream_preserves_sse_and_closes_provider(self):
        response = await self.chat(stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/event-stream", response.headers["content-type"])
        self.assertEqual(response.content, b"".join(self.provider.chunks))
        self.assertTrue(self.provider.closed)

    async def test_stream_failure_before_first_chunk_is_http_error(self):
        self.provider.error = RuntimeError("synthetic-secret")
        response = await self.chat(stream=True)
        self.assertEqual(response.status_code, 502)
        self.assertIn("application/json", response.headers["content-type"])
        self.assertNotIn("synthetic-secret", response.text)
        self.assertTrue(self.provider.closed)

    async def test_empty_stream_is_http_error(self):
        self.provider.chunks = []
        response = await self.chat(stream=True)
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["error"]["code"], "invalid_provider_response")

    async def test_partial_stream_failure_is_sanitized_without_done(self):
        self.provider.fail_after = 1
        response = await self.chat(stream=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Hello", response.text)
        self.assertIn('"error"', response.text)
        self.assertNotIn("[DONE]", response.text)
        self.assertNotIn("synthetic-secret", response.text)
        self.assertTrue(self.provider.closed)

    async def test_closing_stream_iterator_releases_provider(self):
        response = await main.chat(main.ChatRequest(model="test-model", messages=MESSAGES, stream=True))
        await anext(response.body_iterator)
        self.assertFalse(self.provider.closed)
        await response.body_iterator.aclose()
        self.assertTrue(self.provider.closed)

    async def test_unsupported_adapter_controls_return_422_on_both_chat_paths(self):
        for adapter in (providers.AnthropicProvider(), providers.OllamaProvider()):
            providers.PROVIDERS[adapter.name] = adapter
            for stream in (False, True):
                with self.subTest(provider=adapter.name, stream=stream):
                    with patch.object(providers.cfg, "anthropic_key", "synthetic-test-key"), patch.object(
                        providers.httpx, "AsyncClient", side_effect=AssertionError("Unexpected network client"),
                    ):
                        response = await self.chat(model=f"{adapter.name}/test-model", tool_choice="auto", stream=stream)
                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()["error"]["code"], "unsupported_parameter")
