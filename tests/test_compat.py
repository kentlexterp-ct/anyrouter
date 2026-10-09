import json
import unittest
from unittest.mock import patch

import httpx

from tests.support import MESSAGES, providers, main
from anyrouter.config import cfg
from anyrouter.errors import GatewayError


TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Lookup a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
CALL = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": '{"city":"Manila"}'}}
CONVERSATION = [*MESSAGES, {"role": "assistant", "content": None, "tool_calls": [CALL]}, {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"}]
FORMAT = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "additionalProperties": False, "required": ["answer"]}, "strict": True}}


class CompatibilityTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.response = httpx.Response(200, json={"message": {"content": "Hello"}})
        self.responses = []
        original = httpx.AsyncClient
        def handler(request):
            self.requests.append(request)
            return self.responses.pop(0) if self.responses else self.response
        def factory(*args, **kwargs):
            return original(*args, transport=httpx.MockTransport(handler), **kwargs)
        self.network = patch.object(providers.httpx, "AsyncClient", side_effect=factory)
        self.network.start()
        self.addCleanup(self.network.stop)
        self.settings = patch.multiple(cfg, anthropic_key="synthetic-upstream", openrouter_key="synthetic-upstream", openai_key="synthetic-upstream", allow_paid_routing=False)
        self.settings.start()
        self.addCleanup(self.settings.stop)

    async def test_anthropic_tools_results_stop_and_schema_translation(self):
        self.response = httpx.Response(200, json={"id": "msg-test", "model": "known", "content": [{"type": "tool_use", "id": "call-2", "name": "lookup", "input": {"city": "Cebu"}}], "stop_reason": "tool_use", "usage": {"input_tokens": 10, "cache_read_input_tokens": 3, "output_tokens": 4}})
        result = await providers.AnthropicProvider().chat("known", {"messages": CONVERSATION, "tools": TOOLS, "tool_choice": "required", "stop": "END", "response_format": FORMAT})
        payload = json.loads(self.requests[0].content)
        self.assertEqual(payload["messages"][1]["content"][0]["type"], "tool_use")
        self.assertEqual(payload["messages"][2]["content"][0]["tool_use_id"], "call-1")
        self.assertEqual(payload["tool_choice"], {"type": "any"})
        self.assertEqual(payload["tools"][0]["input_schema"], TOOLS[0]["function"]["parameters"])
        self.assertEqual(payload["stop_sequences"], ["END"])
        self.assertEqual(payload["output_config"]["format"]["schema"], FORMAT["json_schema"]["schema"])
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(json.loads(choice["message"]["tool_calls"][0]["function"]["arguments"]), {"city": "Cebu"})
        self.assertEqual(result["usage"]["total_tokens"], 17)

    async def test_ollama_tools_results_and_schema_translation(self):
        self.response = httpx.Response(200, json={"message": {"content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": {"city": "Cebu"}}}]}, "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 4})
        result = await providers.OllamaProvider().chat("known", {"messages": CONVERSATION, "tools": TOOLS, "tool_choice": "auto", "response_format": FORMAT})
        payload = json.loads(self.requests[0].content)
        self.assertEqual(payload["messages"][1]["tool_calls"][0]["function"]["arguments"], {"city": "Manila"})
        self.assertEqual(payload["messages"][2]["tool_name"], "lookup")
        self.assertEqual(payload["tools"], TOOLS)
        self.assertEqual(payload["format"], FORMAT["json_schema"]["schema"])
        self.assertEqual(result["usage"]["total_tokens"], 14)
        self.assertEqual(result["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(result["choices"][0]["message"]["tool_calls"][0]["id"].startswith("call_"))

    async def test_inline_images_are_translated_and_remote_images_are_rejected(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}
        messages = [{"role": "user", "content": [{"type": "text", "text": "Describe"}, image]}]
        for adapter in (providers.AnthropicProvider(), providers.OllamaProvider()):
            self.response = httpx.Response(200, json={"model": "known", "content": [{"type": "text", "text": "Hello"}]} if adapter.name == "anthropic" else {"message": {"content": "Hello"}})
            await adapter.chat("known", {"messages": messages})
            payload = json.loads(self.requests[-1].content)
            if adapter.name == "anthropic":
                self.assertEqual(payload["messages"][0]["content"][1]["source"]["data"], "aW1hZ2U=")
            else:
                self.assertEqual(payload["messages"][0]["images"], ["aW1hZ2U="])
            image["image_url"]["url"] = "http://169.254.169.254/private"
            with self.assertRaises(GatewayError) as caught:
                await adapter.chat("known", {"messages": messages})
            self.assertEqual(caught.exception.status_code, 422)
            image["image_url"]["url"] = "data:image/png;base64,aW1hZ2U="
        self.assertEqual(len(self.requests), 2)

    async def test_tool_argument_validation_and_unsupported_strict_tools(self):
        for adapter in (providers.AnthropicProvider(), providers.OllamaProvider()):
            for arguments_ in ("broken", "[]", "null"):
                call = {**CALL, "function": {**CALL["function"], "arguments": arguments_}}
                with self.assertRaises(GatewayError) as caught:
                    await adapter.chat("known", {"messages": [{"role": "assistant", "content": None, "tool_calls": [call]}]})
                self.assertEqual(caught.exception.status_code, 422)
            with self.assertRaises(GatewayError):
                await adapter.chat("known", {"messages": MESSAGES, "tools": [{"type": "function", "function": {**TOOLS[0]["function"], "strict": True}}]})
        self.assertFalse(self.requests)

    async def test_anthropic_stream_tool_arguments_are_incremental_and_round_trip(self):
        events = [
            {"type": "message_start", "message": {"model": "known", "usage": {"input_tokens": 10, "cache_read_input_tokens": 2, "output_tokens": 0}}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "call-test", "name": "lookup", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"city":'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"Manila"}'}},
            {"type": "content_block_stop", "index": 1},
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
            {"type": "message_stop"},
        ]
        self.response = httpx.Response(200, text="".join("data: " + json.dumps(event) + "\n\n" for event in events))
        chunks = [chunk async for chunk in providers.AnthropicProvider().stream_chat("known", {"messages": MESSAGES, "tools": TOOLS})]
        records = [json.loads(chunk.decode()[6:]) for chunk in chunks[:-1]]
        calls = [record["choices"][0]["delta"]["tool_calls"][0] for record in records if record["choices"][0]["delta"].get("tool_calls")]
        self.assertEqual({call["index"] for call in calls}, {0})
        self.assertEqual(json.loads("".join(call["function"]["arguments"] for call in calls)), {"city": "Manila"})
        self.assertEqual(records[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(records[-1]["usage"]["total_tokens"], 16)
        self.assertEqual(chunks[-1], b"data: [DONE]\n\n")

    async def test_incomplete_or_malformed_tool_stream_never_emits_done(self):
        for partial, close in (("broken", True), ('{"a":1}', False)):
            events = [{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call-test", "name": "lookup", "input": {}}}, {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": partial}}]
            if close:
                events.append({"type": "content_block_stop", "index": 0})
            events.append({"type": "message_stop"})
            self.response = httpx.Response(200, text="".join("data: " + json.dumps(event) + "\n\n" for event in events))
            chunks = []
            with self.assertRaises(GatewayError):
                async for chunk in providers.AnthropicProvider().stream_chat("known", {"messages": MESSAGES, "tools": TOOLS}):
                    chunks.append(chunk)
            self.assertNotIn(b"data: [DONE]\n\n", chunks)

    async def test_zero_argument_tool_stream_emits_valid_object(self):
        events = [{"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call-test", "name": "lookup", "input": {}}}, {"type": "content_block_stop", "index": 0}, {"type": "message_stop"}]
        self.response = httpx.Response(200, text="".join("data: " + json.dumps(event) + "\n\n" for event in events))
        chunks = [chunk async for chunk in providers.AnthropicProvider().stream_chat("known", {"messages": MESSAGES, "tools": TOOLS})]
        records = [json.loads(chunk.decode()[6:]) for chunk in chunks[:-1]]
        arguments = "".join(record["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] for record in records if record["choices"][0]["delta"].get("tool_calls"))
        self.assertEqual(json.loads(arguments), {})

    async def test_http_200_error_bodies_and_invalid_completions_are_sanitized(self):
        for adapter in (providers.OpenAIProvider(), providers.OpenRouterProvider()):
            for response in ({"error": {"message": "synthetic-secret"}}, {"choices": []}, {"choices": [{"message": "synthetic-secret"}]}):
                self.response = httpx.Response(200, json=response)
                with self.assertRaises(GatewayError) as caught:
                    await adapter.chat("known", {"messages": MESSAGES})
                self.assertEqual(caught.exception.status_code, 502)
                self.assertNotIn("synthetic-secret", str(caught.exception.payload()))

    async def test_ollama_stream_tools_and_usage_option(self):
        event = {"message": {"content": "", "tool_calls": [{"function": {"name": "lookup", "arguments": {"city": "Manila"}}}]}, "done": True, "prompt_eval_count": 10, "eval_count": 3}
        for include_usage in (True, False):
            self.response = httpx.Response(200, text=json.dumps(event) + "\n")
            chunks = [chunk async for chunk in providers.OllamaProvider().stream_chat("known", {"messages": MESSAGES, "tools": TOOLS, "stream_options": {"include_usage": include_usage}})]
            records = [json.loads(chunk.decode()[6:]) for chunk in chunks[:-1]]
            self.assertEqual(records[0]["choices"][0]["delta"]["tool_calls"][0]["index"], 0)
            self.assertEqual(records[-1]["choices"][0]["finish_reason"], "tool_calls")
            self.assertEqual("usage" in records[-1], include_usage)

    async def test_openai_compatible_format_and_stream_options_are_forwarded(self):
        self.response = httpx.Response(200, text='data: {"choices":[]}\n\ndata: [DONE]\n\n')
        for adapter in (providers.OpenAIProvider(), providers.OpenRouterProvider()):
            chunks = [chunk async for chunk in adapter.stream_chat("known", {"messages": MESSAGES, "response_format": FORMAT, "stream_options": {"include_usage": True}})]
            payload = json.loads(self.requests[-1].content)
            self.assertEqual(payload["response_format"], FORMAT)
            self.assertEqual(payload["stream_options"], {"include_usage": True})
            self.assertEqual(chunks[-1], b"data: [DONE]\n\n")

    async def test_openrouter_automatic_routing_restricts_price_and_fallbacks(self):
        self.response = httpx.Response(200, json={"choices": [{"message": {"content": "Hello"}}]})
        await providers.OpenRouterProvider().chat("known", {"messages": MESSAGES, "_automatic": True})
        policy = json.loads(self.requests[-1].content)["provider"]
        self.assertFalse(policy["allow_fallbacks"])
        self.assertTrue(policy["require_parameters"])
        self.assertTrue(all(price == 0 for price in policy["max_price"].values()))

    async def test_anthropic_discovery_paginates_and_uses_server_metadata(self):
        self.responses = [httpx.Response(200, json={"data": [{"id": "first", "max_input_tokens": 8192, "capabilities": {"image_input": {"supported": True}}}], "has_more": True, "last_id": "first"}), httpx.Response(200, json={"data": [{"id": "second"}], "has_more": False})]
        catalog = await providers.AnthropicProvider().list_catalog()
        self.assertEqual([r["model"] for r in catalog], ["first", "second"])
        self.assertEqual(self.requests[-1].url.params["after_id"], "first")
        self.assertIn("vision", catalog[0]["capabilities"])
        self.assertNotIn("tools", catalog[0]["capabilities"])
        self.assertIsNone(catalog[1]["context"])

    async def test_openrouter_catalog_does_not_treat_suffix_as_price_evidence(self):
        self.response = httpx.Response(200, json={"data": [{"id": "synthetic:free", "context_length": 8192, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}, "supported_parameters": ["tools", "tool_choice"], "pricing": {"prompt": "0", "completion": "0", "request": "1"}}, {"id": "unknown:free"}]})
        catalog = await providers.OpenRouterProvider().list_catalog()
        self.assertFalse(catalog[0]["free"])
        self.assertIsNone(catalog[1]["free"])
        self.assertIn("tools", catalog[0]["capabilities"])

    async def test_ollama_discovery_handles_unknown_show_capabilities(self):
        self.responses = [httpx.Response(200, json={"models": [{"name": "known"}, {"name": "old-server"}]}), httpx.Response(200, json={"capabilities": ["completion", "tools"], "parameters": "num_ctx 8192\n", "model_info": {"synthetic.context_length": 131072}}), httpx.Response(404)]
        catalog = await providers.OllamaProvider().list_catalog()
        self.assertIn("tools", catalog[0]["capabilities"])
        self.assertEqual(catalog[0]["context"], 8192)
        self.assertNotIn("capabilities", catalog[1])

    async def test_ollama_training_context_is_not_assumed_to_be_serving_context(self):
        self.responses = [httpx.Response(200, json={"models": [{"name": "known"}]}), httpx.Response(200, json={"capabilities": ["completion"], "model_info": {"synthetic.context_length": 131072}})]
        catalog = await providers.OllamaProvider().list_catalog()
        self.assertIsNone(catalog[0]["context"])

    async def test_missing_usage_is_omitted_and_length_reason_is_preserved(self):
        for adapter, response in ((providers.AnthropicProvider(), {"content": [{"type": "text", "text": "Hello"}], "stop_reason": "max_tokens"}), (providers.OllamaProvider(), {"message": {"content": "Hello"}, "done_reason": "length"})):
            self.response = httpx.Response(200, json=response)
            result = await adapter.chat("known", {"messages": MESSAGES})
            self.assertNotIn("usage", result)
            self.assertEqual(result["choices"][0]["finish_reason"], "length")

    def test_chat_request_typed_options_validation(self):
        for invalid in ({"stream_options": {"include_usage": True}}, {"routing": {"input_tokens": 0}}, {"routing": {"input_tokens": True}}, {"max_tokens": True}, {"messages": [{"role": {}, "content": "Hello"}]}, {"tool_choice": {"type": "function", "function": "invalid"}}, {"response_format": {"type": "json_schema", "json_schema": {"name": "x"}}}):
            with self.assertRaises(ValueError):
                main.ChatRequest(**{"model": "known", "messages": MESSAGES, **invalid})
