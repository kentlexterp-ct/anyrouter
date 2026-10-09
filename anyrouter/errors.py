import httpx
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


class GatewayError(Exception):
    """A public error whose message contains no upstream exception details."""

    def __init__(self, status_code, code, message, provider=None, retry_after=None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.provider = provider
        self.retry_after = retry_after

    def payload(self):
        error = {"code": self.code, "message": self.message}
        if self.provider:
            error["provider"] = self.provider
        return {"error": error}


def provider_error(provider, exc):
    if isinstance(exc, GatewayError):
        return exc
    if isinstance(exc, httpx.TimeoutException):
        return GatewayError(504, "provider_timeout", "Provider request timed out.", provider)
    if isinstance(exc, httpx.ConnectError):
        return GatewayError(502, "provider_connect_error", "Provider connection failed.", provider)
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:
            header = exc.response.headers.get("Retry-After", "0")
            try:
                retry = float(header)
            except ValueError:
                try:
                    retry = (parsedate_to_datetime(header) - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    retry = 0
            retry = min(3600, max(0, retry))
            return GatewayError(429, "provider_rate_limit", "Provider rate limit reached.", provider, retry)
        if status in (401, 403):
            return GatewayError(502, "provider_auth_error", "Provider credentials were rejected.", provider)
        if status in (400, 404, 422):
            return GatewayError(status, "provider_rejected_request", "Provider rejected the request.", provider)
    return GatewayError(502, "provider_error", "Provider request failed.", provider)
