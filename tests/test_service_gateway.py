"""HTTP contract tests for the edge-service FastAPI gateway."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import httpx

from service_edge_inference.domain import ServiceModelConfig
from service_edge_inference.gateway import GatewaySettings, create_app
from service_edge_inference.llama_client import LlamaServerUnavailable

REPO_ROOT = Path(__file__).resolve().parent.parent


class RecordingTokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
        return messages[-1]["content"]


class FakeLlamaServer:
    def __init__(self) -> None:
        self.healthy = True
        self.completion_output = "b"
        self.fail_warm = False
        self.fail_classify = False
        self.requests: list[dict[str, Any]] = []
        self.closed = False

    async def health(self) -> bool:
        return self.healthy

    async def completion(
        self, *, prompt: str, grammar: str, n_predict: int, cache_prompt: bool = True
    ) -> str:
        self.requests.append(
            {
                "prompt": prompt,
                "grammar": grammar,
                "n_predict": n_predict,
                "cache_prompt": cache_prompt,
            }
        )
        if n_predict == 0 and self.fail_warm:
            raise LlamaServerUnavailable("warm-up failed")
        if n_predict == 1 and self.fail_classify:
            raise LlamaServerUnavailable("inference failed")
        return "" if n_predict == 0 else self.completion_output

    async def close(self) -> None:
        self.closed = True


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        model_config = ServiceModelConfig.load(
            REPO_ROOT / "service_edge_inference" / "model_selection_matrix.json",
            REPO_ROOT / "manifests" / "v2.1.json",
        )
        self.llama = FakeLlamaServer()
        self.app = create_app(
            model_config=model_config,
            tokenizer=RecordingTokenizer(),
            llama_server=self.llama,
            settings=GatewaySettings(admin_token="test-admin-token"),
        )
        self.transport = httpx.ASGITransport(app=self.app)

    async def asyncTearDown(self) -> None:
        await self.transport.aclose()

    async def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self.transport, base_url="http://gateway.test")

    async def test_liveness_auth_warm_update_and_classification_contract(self) -> None:
        client = await self._client()
        self.assertEqual((await client.get("/health/live")).status_code, 200)
        self.assertEqual((await client.get("/health/ready")).status_code, 503)
        self.assertEqual((await client.post("/v1/classify", json={"query": "route me"})).status_code, 503)

        tools = [
            {"name": "call_handler", "description": "Call a registered handler."},
            {"name": "nav_route_planner", "description": "Plan a driving route."},
        ]
        self.assertEqual((await client.put("/v1/toolset", json=tools)).status_code, 401)

        update = await client.put(
            "/v1/toolset", json=tools, headers={"X-Admin-Token": "test-admin-token"}
        )
        self.assertEqual(update.status_code, 200)
        self.assertTrue(update.json()["warm"])
        self.assertEqual(self.llama.requests[0]["n_predict"], 0)
        self.assertEqual(self.llama.requests[0]["grammar"], 'root ::= "a" | "b" | "-"\n')

        classify = await client.post("/v1/classify", json={"query": "Please plan a route."})
        self.assertEqual(classify.status_code, 200)
        self.assertEqual(classify.json()["tool_id"], "b")
        self.assertEqual(classify.json()["tool"], "nav_route_planner")
        self.assertEqual(self.llama.requests[1]["n_predict"], 1)
        self.assertEqual((await client.get("/health/ready")).status_code, 200)
        await client.aclose()

    async def test_invalid_model_output_becomes_none_but_engine_failure_is_503(self) -> None:
        client = await self._client()
        tools = [{"name": "call_handler", "description": "Call a registered handler."}]
        await client.put(
            "/v1/toolset", json=tools, headers={"X-Admin-Token": "test-admin-token"}
        )

        self.llama.completion_output = "z"
        invalid = await client.post("/v1/classify", json={"query": "Call support."})
        self.assertEqual(invalid.status_code, 200)
        self.assertEqual(invalid.json()["tool_id"], "-")
        self.assertEqual(invalid.json()["tool"], "none")

        self.llama.healthy = False
        self.assertEqual((await client.get("/health/ready")).status_code, 503)

        self.llama.fail_classify = True
        engine_failure = await client.post("/v1/classify", json={"query": "Call support."})
        self.assertEqual(engine_failure.status_code, 503)
        await client.aclose()

    async def test_failed_warm_up_preserves_the_active_tool_set(self) -> None:
        client = await self._client()
        first_tools = [{"name": "call_handler", "description": "Call a registered handler."}]
        headers = {"X-Admin-Token": "test-admin-token"}
        first = await client.put("/v1/toolset", json=first_tools, headers=headers)
        first_version = first.json()["toolset_version"]

        self.llama.fail_warm = True
        failed = await client.put(
            "/v1/toolset",
            json=[{"name": "route_planner", "description": "Plan a driving route."}],
            headers=headers,
        )
        self.assertEqual(failed.status_code, 503)

        current = await client.get("/v1/toolset/status", headers=headers)
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["toolset_version"], first_version)
        self.assertEqual(current.json()["tool_count"], 1)
        await client.aclose()


if __name__ == "__main__":
    unittest.main()
