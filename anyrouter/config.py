import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    host: str = os.getenv("ANYROUTER_HOST", "127.0.0.1")
    port: int = int(os.getenv("ANYROUTER_PORT", "8000"))
    openai_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    google_key: str | None = os.getenv("GOOGLE_API_KEY")
    openrouter_key: str | None = os.getenv("OPENROUTER_API_KEY")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def has_openai(self):
        return bool(self.openai_key)

    @property
    def has_anthropic(self):
        return bool(self.anthropic_key)

    @property
    def has_google(self):
        return bool(self.google_key)

    @property
    def has_openrouter(self):
        return bool(self.openrouter_key)

cfg = Config()
