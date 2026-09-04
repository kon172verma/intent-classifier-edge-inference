"""Small asynchronous client for the internal llama-server HTTP API."""

from __future__ import annotations

from typing import Any

import httpx


class LlamaServerError(RuntimeError):
    """Base error for unavailable or invalid llama-server responses."""


class LlamaServerUnavailable(LlamaServerError):
    """The inference server could not be reached or returned a failing status."""


class LlamaServerProtocolError(LlamaServerError):
    """The inference server returned a response outside the expected schema."""


class LlamaServerClient:
    """Internal-only llama-server client with explicit timeout and schema checks."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
        )

    async def health(self) -> bool:
        """Return whether llama-server reports a successful health response."""
        try:
            response = await self._client.get("/health")
        except httpx.HTTPError as exc:
            raise LlamaServerUnavailable("Unable to reach llama-server health endpoint") from exc
        return response.is_success

    async def completion(
        self,
        *,
        prompt: str,
        grammar: str,
        n_predict: int,
        cache_prompt: bool = True,
    ) -> str:
        """Request one constrained completion or an empty warm-up completion."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(grammar, str) or not grammar:
            raise ValueError("grammar must be a non-empty string")
        if n_predict < 0:
            raise ValueError("n_predict must not be negative")
        payload = {
            "prompt": prompt,
            "grammar": grammar,
            "n_predict": n_predict,
            "cache_prompt": cache_prompt,
            "temperature": 0,
        }
        try:
            response = await self._client.post("/completion", json=payload)
        except httpx.HTTPError as exc:
            raise LlamaServerUnavailable("Unable to reach llama-server completion endpoint") from exc
        if not response.is_success:
            raise LlamaServerUnavailable(
                f"llama-server completion endpoint returned HTTP {response.status_code}"
            )
        try:
            body: Any = response.json()
        except ValueError as exc:
            raise LlamaServerProtocolError("llama-server completion response is not JSON") from exc
        if not isinstance(body, dict) or not isinstance(body.get("content"), str):
            raise LlamaServerProtocolError(
                "llama-server completion response requires a string content field"
            )
        return body["content"]

    async def close(self) -> None:
        """Close the owned HTTP client; injected clients remain caller-owned."""
        if self._owns_client:
            await self._client.aclose()
