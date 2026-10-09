import os
from unittest.mock import patch

import dotenv

# Import configuration without loading real credentials or reading .env.
with patch.object(dotenv, "load_dotenv", return_value=False), patch.dict(os.environ, {}, clear=True):
    from anyrouter import main, providers


MESSAGES = [{"role": "user", "content": "Hello"}]


class FakeProvider:
    available = True

    def __init__(self):
        self.result = {"choices": [{"message": {"content": "warm"}}]}
        self.error = None
        self.calls = []
        self.chunks = [b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n', b"data: [DONE]\n\n"]
        self.fail_after = None
        self.closed = False

    async def list_models(self):
        return ["test-model"]

    async def chat(self, model, body):
        self.calls.append((model, body))
        if self.error:
            raise self.error
        return self.result

    async def stream_chat(self, model, body):
        self.calls.append((model, body))
        try:
            if self.error:
                raise self.error
            for index, chunk in enumerate(self.chunks):
                if index == self.fail_after:
                    raise RuntimeError("synthetic-secret must not reach clients")
                yield chunk
        finally:
            self.closed = True
