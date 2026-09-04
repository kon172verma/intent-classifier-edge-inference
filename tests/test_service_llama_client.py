"""Contract tests for the internal llama-server HTTP client."""

from __future__ import annotations

import json
import unittest

import httpx

from service_edge_inference.llama_client import (
    LlamaServerClient,
    LlamaServerProtocolError,
    LlamaServerUnavailable,
)


class LlamaServerClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_completion_posts_constrained_cacheable_request(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"content": "b"})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://llama.internal"
        )
        llama = LlamaServerClient("http://ignored", client=client)
        completion = await llama.completion(
            prompt="prompt", grammar='root ::= "a" | "-"\n', n_predict=1
        )

        self.assertEqual(completion, "b")
        self.assertEqual(seen["url"], "http://llama.internal/completion")
        self.assertEqual(
            seen["body"],
            {
                "prompt": "prompt",
                "grammar": 'root ::= "a" | "-"\n',
                "n_predict": 1,
                "cache_prompt": True,
                "temperature": 0,
            },
        )
        await client.aclose()

    async def test_protocol_and_http_errors_are_explicit(self) -> None:
        async def assert_failure(response: httpx.Response, error: type[Exception]) -> None:
            client = httpx.AsyncClient(
                transport=httpx.MockTransport(lambda _: response), base_url="http://llama.internal"
            )
            llama = LlamaServerClient("http://ignored", client=client)
            with self.assertRaises(error):
                await llama.completion(prompt="prompt", grammar='root ::= "-"\n', n_predict=1)
            await client.aclose()

        await assert_failure(httpx.Response(503, json={"error": "busy"}), LlamaServerUnavailable)
        await assert_failure(httpx.Response(200, json={"unexpected": "value"}), LlamaServerProtocolError)

    async def test_health_is_false_for_non_success_and_raises_for_transport_error(self) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: httpx.Response(503)),
            base_url="http://llama.internal",
        )
        self.assertFalse(await LlamaServerClient("http://ignored", client=client).health())
        await client.aclose()

        def fail(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        failing_client = httpx.AsyncClient(
            transport=httpx.MockTransport(fail), base_url="http://llama.internal"
        )
        with self.assertRaises(LlamaServerUnavailable):
            await LlamaServerClient("http://ignored", client=failing_client).health()
        await failing_client.aclose()
