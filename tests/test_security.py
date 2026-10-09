import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from tests.support import FakeProvider, MESSAGES, main, providers
from anyrouter.config import cfg
from anyrouter.errors import GatewayError
from anyrouter.observability import metrics
from anyrouter.security import KeyStore, allows


class SecurityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "synthetic-keys.json"
        self.raw = "synthetic-caller-key"
        self.policy = {"sha256": hashlib.sha256(self.raw.encode()).hexdigest(), "scopes": ["chat", "models", "classify", "admin"], "rpm": 100, "daily_requests": 1000}
        self.write({"test": self.policy})
        self.settings = patch.multiple(cfg, key_file=str(self.path), auth_required=False, host="127.0.0.1")
        self.settings.start()
        self.addCleanup(self.settings.stop)
        main.keys.windows.clear()
        main.keys.source = None
        self.provider = FakeProvider()
        self.registry = patch.dict(providers.PROVIDERS, {"ollama": self.provider}, clear=True)
        self.registry.start()
        self.addCleanup(self.registry.stop)
        self.index = patch.dict(main._MODEL_INDEX, {"known": "ollama"}, clear=True)
        self.index.start()
        self.addCleanup(self.index.stop)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test")

    def write(self, keys):
        self.path.write_text(json.dumps(keys), encoding="utf-8")

    async def asyncTearDown(self):
        await self.client.aclose()

    def headers(self, raw=None):
        return {"Authorization": "Bearer " + (self.raw if raw is None else raw)}

    async def chat(self, headers=None, **body):
        return await self.client.post("/v1/chat/completions", json={"model": "ollama:known", "messages": MESSAGES, **body}, headers=headers)

    async def test_missing_invalid_and_duplicate_keys_fail_before_upstream(self):
        for headers in (None, self.headers("wrong"), [("Authorization", "Bearer " + self.raw), ("Authorization", "Bearer " + self.raw)]):
            self.assertEqual((await self.chat(headers=headers)).status_code, 401)
        self.assertFalse(self.provider.calls)

    async def test_caller_key_never_enters_provider_payload(self):
        response = await self.chat(headers=self.headers())
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(self.raw, json.dumps(self.provider.calls))
        self.assertEqual(len(response.headers["x-request-id"]), 32)

    async def test_scopes_and_provider_model_allowlists_apply_to_chat_and_webhook(self):
        self.policy["scopes"] = ["models"]
        self.write({"test": self.policy})
        self.assertEqual((await self.chat(headers=self.headers())).status_code, 403)
        self.assertEqual((await self.client.post("/webhook/classify", json={}, headers=self.headers())).status_code, 403)
        self.policy["scopes"] = ["chat", "models", "classify"]
        self.policy["models"] = ["ollama:other"]
        self.write({"test": self.policy})
        self.assertEqual((await self.chat(headers=self.headers())).status_code, 403)
        self.assertEqual((await self.client.get("/v1/models", headers=self.headers())).json()["data"], [])
        self.assertEqual((await self.client.post("/webhook/classify", json={}, headers=self.headers())).status_code, 403)
        self.assertFalse(self.provider.calls)

    async def test_missing_scopes_grants_no_access(self):
        del self.policy["scopes"]
        self.write({"test": self.policy})
        self.assertEqual((await self.chat(headers=self.headers())).status_code, 403)
        self.assertFalse(self.provider.calls)

    async def test_revocation_rotation_disabled_and_expired_keys(self):
        self.assertEqual((await self.chat(headers=self.headers())).status_code, 200)
        rotated = "rotated-synthetic-key"
        self.policy["sha256"] = hashlib.sha256(rotated.encode()).hexdigest()
        self.write({"test": self.policy})
        self.assertEqual((await self.chat(headers=self.headers())).status_code, 401)
        self.assertEqual((await self.chat(headers=self.headers(rotated))).status_code, 200)
        self.policy["disabled"] = True
        self.write({"test": self.policy})
        self.assertEqual((await self.chat(headers=self.headers(rotated))).status_code, 401)
        self.policy["disabled"] = False
        self.policy["expires_at"] = 1
        self.write({"test": self.policy})
        self.assertEqual((await self.chat(headers=self.headers(rotated))).status_code, 401)
        self.write({})
        self.assertEqual((await self.chat(headers=self.headers(rotated))).status_code, 401)

    async def test_invalid_key_file_fails_closed_and_recovers(self):
        self.assertEqual((await self.chat(headers=self.headers())).status_code, 200)
        self.path.write_text("broken-secret", encoding="utf-8")
        response = await self.chat(headers=self.headers())
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("broken-secret", response.text)
        self.write({"test": self.policy})
        self.assertEqual((await self.chat(headers=self.headers())).status_code, 200)

    async def test_rate_and_daily_quotas_are_distinct_and_survive_rotation(self):
        self.policy["rpm"] = 1
        self.write({"test": self.policy})
        self.assertEqual((await self.chat(headers=self.headers())).status_code, 200)
        response = await self.chat(headers=self.headers())
        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.json()["error"]["code"], "rate_limit")
        self.assertEqual(response.headers["retry-after"], "60")
        self.policy["rpm"] = 100
        self.policy["daily_requests"] = 1
        self.write({"test": self.policy})
        response = await self.chat(headers=self.headers())
        self.assertEqual(response.json()["error"]["code"], "quota_exceeded")
        self.assertEqual(len(self.provider.calls), 1)

    async def test_public_clients_and_public_bind_require_keys_even_without_flag(self):
        with patch.object(cfg, "key_file", None):
            self.assertEqual((await self.chat()).status_code, 200)
            with patch.object(cfg, "host", "0.0.0.0"):
                self.assertEqual((await self.chat()).status_code, 401)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app, client=("203.0.113.1", 10)), base_url="http://test") as remote:
                self.assertEqual((await remote.get("/v1/models")).status_code, 401)
            self.assertEqual((await self.client.get("/metrics")).status_code, 401)

    async def test_public_startup_without_keys_is_refused(self):
        with patch.multiple(cfg, host="0.0.0.0", key_file=None):
            with self.assertRaises(RuntimeError):
                async with main.lifespan(main.app):
                    self.fail("Startup must fail")

    async def test_root_path_does_not_bypass_authentication(self):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app, root_path="/gateway"), base_url="http://test/gateway") as client:
            response = await client.post("/v1/chat/completions", json={"model": "ollama:known", "messages": MESSAGES})
            self.assertEqual(response.status_code, 401)
        self.assertFalse(self.provider.calls)

    async def test_body_limit_rejects_without_upstream_and_errors_have_ids(self):
        with patch.object(cfg, "max_body_bytes", 32):
            response = await self.chat(headers=self.headers())
        self.assertEqual(response.status_code, 413)
        self.assertIn("x-request-id", response.headers)
        self.assertFalse(self.provider.calls)

    async def test_logs_and_metrics_exclude_prompts_credentials_and_freeform_paths(self):
        secret = "synthetic-private-email@example.test"
        with self.assertLogs("anyrouter.requests", level="INFO") as logs:
            response = await self.chat(headers=self.headers(), messages=[{"role": "user", "content": secret}])
            await self.client.get("/" + secret)
        self.assertEqual(response.status_code, 200)
        output = "\n".join(logs.output)
        self.assertNotIn(secret, output)
        self.assertNotIn(self.raw, output)
        self.assertIn(response.headers["x-request-id"], output)
        exported = await self.client.get("/metrics", headers=self.headers())
        self.assertEqual(exported.status_code, 200)
        self.assertNotIn(secret, exported.text)
        self.assertNotIn(self.raw, exported.text)
        self.assertIn("anyrouter_requests_total", exported.text)

    async def test_stream_errors_are_logged_despite_http_200(self):
        self.provider.fail_after = 1
        with self.assertLogs("anyrouter.requests", level="INFO") as logs:
            response = await self.chat(headers=self.headers(), stream=True)
        self.assertEqual(response.status_code, 200)
        record = json.loads(logs.records[-1].getMessage())
        self.assertEqual(record["error"], "provider_error")


class KeyPolicyTests(unittest.TestCase):
    def test_invalid_and_duplicate_policies_are_refused(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(cfg, "key_file", str(Path(directory) / "keys.json")):
            path = Path(cfg.key_file)
            digest = "0" * 64
            for content in ([{}], {"a": {"sha256": digest, "scopes": ["unknown"]}}, {"a": {"sha256": digest, "rpm": 0}}, {"a": {"sha256": digest}, "b": {"sha256": digest}}, {"a": {"sha256": digest, "daily_tokens": 10}}):
                path.write_text(json.dumps(content), encoding="utf-8")
                with self.assertRaises(GatewayError):
                    KeyStore().load()

    def test_local_only_rejects_remote_ollama_and_remote_providers(self):
        policy = {"local_only": True}
        self.assertFalse(allows(policy, "openai", "known"))
        with patch.object(cfg, "ollama_url", "http://10.0.0.1:11434"):
            self.assertFalse(allows(policy, "ollama", "known"))
