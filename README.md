# AnyRouter V2

OpenAI-compatible gateway with secure multi-provider routing

## Architecture

The `anyrouter/` package is organized into focused, independent modules:

- `anyrouter.main` — ASGI entry point and lifespan management
- `anyrouter.routing` — automatic routing, cooldown, single-probe recovery, error categories
- `anyrouter.providers` — OpenAI, Anthropic, Ollama, OpenRouter adapters
- `anyrouter.streaming` — Server-Sent Events handling and framing
- `anyrouter.transport` — HTTPX client management, connection pooling, TLS verification
- `anyrouter.security` — authentication, key policies, rate limiting
- `anyrouter.observability` — metrics, request tracing, health checks
- `anyrouter.config` — environment variable configuration, validation
- `anyrouter.errors` — error code mapping and sanitization
- `anyrouter.compat` — OpenAI API parity layer

## Supported Models & Providers

- **OpenAI**: `openai/gpt-4`, `openai/gpt-3.5-turbo`, and compatible models
- **Anthropic**: `anthropic/claude-3-opus-20240229`, `anthropic/claude-3-sonnet-20240229`
- **Ollama**: `ollama:llama3.2:latest`, `ollama:codegemma`, etc. Local-only by default
- **OpenRouter**: `openrouter:model` — free tier and paid models with routing controls

## Setup & Installation

From the repository root (Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e .
.\.venv\Scripts\python.exe -m anyrouter.main
```

Configuration uses environment variables and optional `.env` loading:

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`
- `ANYROUTER_HOST`, `ANYROUTER_PORT`

Keep credentials out of version control.

## API Examples

**curl:**

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_KEY" \
  -d '{"model": "ollama/llama3.2:latest", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Python client:**

```python
import requests

resp = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer KEY",
    },
    json={"model": "ollama/llama3.2:latest",
          "messages": [{"role": "user", "content": "Hello"}]},
)
```

## Authentication

Caller keys are managed via `ANYROUTER_KEY_FILE`. The gateway reads the key file and never writes it. Supported scopes: `chat`, `models`, `classify`, `admin` (`/metrics`). Keys are sent as `Authorization: Bearer KEY`. Invalid or unreadable key files fail closed with HTTP 503. Keys are independent of upstream provider credentials and are never forwarded upstream.

Supported scopes grant nothing if omitted; empty provider/model lists allow all otherwise permitted targets. Model lists use canonical `provider:model` IDs and also constrain catalog visibility and the classifier.

## Automatic Routing + Fallback + Circuit Breakers

`anyrouter/routing.py` implements:

- **Cooldown**: `ANYROUTER_BREAKER_COOLDOWN` (30s default) after consecutive failures; upstream `Retry-After` can extend it
- **Single-probe recovery**: After cooldown, only one half-open circuit is admitted; late in-flight successes cannot close a newer open circuit
- **Error categories**: Automatic routing permits retry for connection failures, circuit cooldown, local admission saturation, or upstream 429. Timeouts, authentication failures, invalid requests, and ambiguous server errors are never replayed. Explicit requests never switch providers. All attempts share one overall deadline; output is never replayed after streaming starts.
- **Circuit breaker threshold**: `ANYROUTER_BREAKER_THRESHOLD` (5 consecutive failures before opening a circuit)
- **Routing attempts**: `ANYROUTER_ROUTING_ATTEMPTS` (max 1–4 eligible candidates tried; range configured via env var)

## Streaming

`anyrouter/streaming.py` handles Server-Sent Events:

- The gateway reads the first provider chunk before returning HTTP 200
- Initial provider failures return a JSON error with HTTP 502/504 (or 422 for unsupported controls)
- After output starts, failures produce an SSE `data:` record containing the sanitized `error` object and close the stream without a synthetic `[DONE]`
- HTTP status cannot be changed after streaming begins
- OpenAI/OpenRouter successful event payloads retain upstream fields (tool deltas, usage, finish reasons); framing is normalized and comment-only keepalives are omitted
- Anthropic and Ollama chunks include a stable completion ID, model, timestamp, and `chat.completion.chunk` object type
- `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers
- Disconnects cancel pending generation before the first output and during streaming; cleanup closes response iterators and releases concurrency slots

## Rate Limiting

Configuration via `anyrouter/security.py`:

- `ANYROUTER_MAX_CONCURRENCY` (8 default): active chat/classification requests per provider, per worker; streams retain their slot until closed
- `ANYROUTER_MAX_REQUESTS` (64 default): total concurrent HTTP requests per worker, including body reads and streams
- `ANYROUTER_MAX_BODY_BYTES` (1048576 default): maximum buffered request body; excess returns 413
- `ANYROUTER_ANONYMOUS_RPM` (60 default): shared loopback development rate limit, daily quota 10000
- Per-key `rpm` and `daily_requests` policies from the key file
- Body reads have an overall request deadline; returns 408 on expiry
- Invalid keys/scopes return 401/403; caller rate/daily quota exhaustion returns 429

## Monitoring & Observability

`anyrouter/observability.py` provides:

- `X-Request-ID` generated on every HTTP response
- Structured request logs containing only route category, status, request ID, provider, error code, and elapsed time (bodies, prompts, headers, and arbitrary paths/model names are excluded)
- `/` — liveness summary (public endpoint)
- `/ready` — returns 200 when at least one configured provider has fresh successful discovery and an admissible circuit, otherwise 503
- `/metrics` — requires admin-scoped key; exports Prometheus text format with bounded provider/route/status labels, request/upstream counters, elapsed sums, observed token totals, and circuit states. No collector or paid service is required; missing upstream usage stays uncounted

## Testing

```powershell
python -B -m unittest discover -s tests -v
```

Current suite: 125 passed + 89 subtests, 21 routing tests passed. Tests cover route compatibility, classification fallback, adapter contracts, catalog discovery, routing policies, breaker races/recovery, authentication, key rotation/revocation, quotas, log redaction, stream framing/tool arguments, client reuse/shutdown, deadlines, admission limits, and disconnectes. Security/bootstrap tests create only synthetic files in temporary directories.

## Limitations

Live cross-provider failover and production deployment have NOT been verified. Everything else is verified via tests and simulated failover. Paid routing requires explicit operator opt-in (`ANYROUTER_ALLOW_PAID_ROUTING`). Remote Ollama deployments need verified pricing metadata. Model capabilities, context budgets, and cost policies must be configured via overrides for AUTO routing. Explicit requests never switch providers automatically.