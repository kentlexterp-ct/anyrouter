import json
import unittest
from unittest.mock import patch

import httpx

from anyrouter.errors import GatewayError
from tests.support import MESSAGES, providers


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.response = httpx.Response(200, json={"message": {"content": "Hello"}})
        original_client = httpx.AsyncClient

        def handler(request):
            self.requests.append(request)
            return self.response

        def factory(*args, **kwargs):
            return original_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        self.client_patch = patch.object(providers.httpx, "AsyncClient", side_effect=factory)
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        for field in ("openai_key", "anthropic_key", "openrouter_key"):
            key_patch = patch.object(providers.cfg, field, "synthetic-test-key")
            key_patch.start()
            self.addCleanup(key_patch.stop)

    async def test_anthropic_chat_uses_resolved_model_and_keeps_system_prompt(self):
        body = {"model": "anthropic/test-model", "messages": [{"role": "system", "content": "Be helpful"}, *MESSAGES], "max_tokens": 10}
        self.response = httpx.Response(200, json={"id": "test-id", "model": "test-model", "content": [{"type": "text", "text": "Hello"}]})
        result = await providers.AnthropicProvider().chat("test-model", body)
        sent = json.loads(self.requests[0].content)
        self.assertEqual(sent["model"], "test-model")
        self.assertEqual(sent["system"], "Be helpful")
        self.assertEqual(sent["messages"], MESSAGES)
        self.assertEqual(sent["max_tokens"], 10)
        self.assertEqual(body["model"], "anthropic/test-model")
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello")

    async def test_anthropic_stream_uses_resolved_model(self):
        self.response = httpx.Response(200, text='data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hello"}}\n\ndata: {"type":"message_stop"}\n\n')
        chunks = [chunk async for chunk in providers.AnthropicProvider().stream_chat("test-model", {"model": "anthropic/test-model", "messages": MESSAGES})]
        self.assertEqual(json.loads(self.requests[0].content)["model"], "test-model")
        self.assertIn(b"Hello", chunks[0])
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")

    async def test_anthropic_unsupported_controls_fail_before_network(self):
        for fields in ({"tool_choice": "auto"}, {"response_format": {"type": "json_object"}}, {"temperature": 2}, {"top_p": 0.5, "temperature": 0.5}):
            with self.subTest(fields=fields), self.assertRaises(GatewayError) as caught:
                await providers.AnthropicProvider().chat("test-model", {"messages": MESSAGES, **fields})
            self.assertEqual(caught.exception.status_code, 422)
        self.assertFalse(self.requests)

    async def test_anthropic_stream_errors_and_truncation_never_emit_done(self):
        for text, code in (
            ('data: {"type":"error","error":{"message":"synthetic-secret"}}\n\n', "provider_error"),
            ('data: broken\n\n', "invalid_provider_response"),
            ('data: []\n\n', "invalid_provider_response"),
            ('data: {"type":"ping"}\n\n', "incomplete_stream"),
        ):
            with self.subTest(text=text):
                self.response = httpx.Response(200, text=text)
                chunks = []
                with self.assertRaises(GatewayError) as caught:
                    async for chunk in providers.AnthropicProvider().stream_chat("test-model", {"messages": MESSAGES}):
                        chunks.append(chunk)
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn(b"data: [DONE]\n\n", chunks)
                self.assertNotIn("synthetic-secret", str(caught.exception))

    async def test_ollama_generation_options_reach_upstream(self):
        body = {"messages": MESSAGES, "temperature": 0.1, "top_p": 0.9, "max_tokens": 10, "stop": "END"}
        result = await providers.OllamaProvider().chat("test-model", body)
        sent = json.loads(self.requests[0].content)
        self.assertEqual(sent["options"], {"temperature": 0.1, "top_p": 0.9, "num_predict": 10, "stop": ["END"]})
        self.assertFalse(sent["stream"])
        self.assertEqual(result["choices"][0]["message"]["content"], "Hello")

    async def test_ollama_stream_preserves_controls_and_done(self):
        self.response = httpx.Response(200, text='{"message":{"content":"Hello"},"done":false}\n{"message":{"content":""},"done":true}\n')
        chunks = [chunk async for chunk in providers.OllamaProvider().stream_chat("test-model", {"messages": MESSAGES, "max_tokens": 10, "stop": ["END"]})]
        sent = json.loads(self.requests[0].content)
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["options"], {"num_predict": 10, "stop": ["END"]})
        self.assertIn(b"Hello", chunks[0])
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")

    async def test_ollama_unsupported_tool_choices_are_rejected_before_network(self):
        for fields in ({"tool_choice": "required"}, {"tool_choice": "auto"}):
            with self.subTest(fields=fields), self.assertRaises(GatewayError) as caught:
                await providers.OllamaProvider().chat("test-model", {"messages": MESSAGES, **fields})
            self.assertEqual(caught.exception.status_code, 422)
        self.assertFalse(self.requests)

    async def test_ollama_error_body_is_not_wrapped_as_success(self):
        self.response = httpx.Response(200, json={"error": "synthetic-secret"})
        with self.assertRaises(GatewayError) as caught:
            await providers.OllamaProvider().chat("test-model", {"messages": MESSAGES})
        self.assertEqual(caught.exception.code, "provider_error")
        self.assertNotIn("synthetic-secret", str(caught.exception))

    async def test_ollama_stream_errors_and_truncation_never_emit_done(self):
        for text, code in (
            ('{"error":"synthetic-secret"}\n', "provider_error"),
            ('broken\n', "invalid_provider_response"),
            ('[]\n', "invalid_provider_response"),
            ('{"message":{"content":"Hello"},"done":false}\n', "incomplete_stream"),
        ):
            with self.subTest(text=text):
                self.response = httpx.Response(200, text=text)
                chunks = []
                with self.assertRaises(GatewayError) as caught:
                    async for chunk in providers.OllamaProvider().stream_chat("test-model", {"messages": MESSAGES}):
                        chunks.append(chunk)
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn(b"data: [DONE]\n\n", chunks)

    async def test_openai_and_openrouter_forward_existing_controls(self):
        body = {"model": "gateway/test-model", "messages": MESSAGES, "temperature": 0.2, "max_tokens": 10, "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {}}}], "tool_choice": "auto"}
        for adapter in (providers.OpenAIProvider(), providers.OpenRouterProvider()):
            with self.subTest(provider=adapter.name):
                self.response = httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
                await adapter.chat("test-model", body)
                sent = json.loads(self.requests[-1].content)
                self.assertEqual(sent, {**body, "model": "test-model"})

    async def test_openai_and_openrouter_streams_remain_passthrough(self):
        expected = 'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\ndata: [DONE]\n\n'
        for adapter in (providers.OpenAIProvider(), providers.OpenRouterProvider()):
            with self.subTest(provider=adapter.name):
                self.response = httpx.Response(200, text=expected)
                chunks = [chunk async for chunk in adapter.stream_chat("test-model", {"messages": MESSAGES})]
                self.assertEqual(b"".join(chunks).decode(), expected)
                self.assertTrue(json.loads(self.requests[-1].content)["stream"])

    async def test_provider_http_errors_are_propagated_to_gateway(self):
        self.response = httpx.Response(401, json={"error": "synthetic-secret"})
        for adapter in (providers.OpenAIProvider(), providers.OpenRouterProvider(), providers.AnthropicProvider(), providers.OllamaProvider()):
            with self.subTest(provider=adapter.name), self.assertRaises(httpx.HTTPStatusError):
                await adapter.chat("test-model", {"messages": MESSAGES})
