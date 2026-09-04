"""FastAPI gateway for the fixed-model edge tool-routing service."""

from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from pydantic import BaseModel, ConfigDict, Field

from service_edge_inference.domain import (
    MAX_POSITIONAL_TOOLS,
    ClassificationResult,
    ServiceModelConfig,
    ToolConfig,
    ToolSetValidationError,
    classify_completion,
    render_full_prompt,
    render_warm_prompt,
)
from service_edge_inference.llama_client import LlamaServerError


class LlamaServerProtocol(Protocol):
    async def health(self) -> bool: ...

    async def completion(
        self, *, prompt: str, grammar: str, n_predict: int, cache_prompt: bool = True
    ) -> str: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class GatewaySettings:
    """Gateway-only limits and administrative access configuration."""

    admin_token: str
    max_tools: int = MAX_POSITIONAL_TOOLS

    def __post_init__(self) -> None:
        if not self.admin_token:
            raise ValueError("admin_token must not be empty")
        if not 1 <= self.max_tools <= MAX_POSITIONAL_TOOLS:
            raise ValueError(f"max_tools must be between 1 and {MAX_POSITIONAL_TOOLS}")


class GatewayNotReady(RuntimeError):
    """No successfully warmed tool set is currently active."""


class ToolInput(BaseModel):
    """Administrative tool-set member; order is supplied by the request list."""

    model_config = ConfigDict(extra="forbid")
    name: str
    description: str


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1)


class ClassificationResponse(BaseModel):
    tool_id: str
    tool: str
    toolset_version: str


class ToolSetStatusResponse(BaseModel):
    toolset_version: str | None
    tool_count: int
    warm: bool


class GatewayState:
    """Serializes inference and tool-set warm-up for llama-server --parallel 1."""

    def __init__(
        self,
        *,
        model_config: ServiceModelConfig,
        tokenizer: Any,
        llama_server: LlamaServerProtocol,
        settings: GatewaySettings,
        initial_tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        import asyncio

        self.model_config = model_config
        self.tokenizer = tokenizer
        self.llama_server = llama_server
        self.settings = settings
        self._lock = asyncio.Lock()
        self._active: ToolConfig | None = (
            ToolConfig.from_records(
                initial_tools,
                model_config=model_config,
                max_tools=settings.max_tools,
            )
            if initial_tools is not None
            else None
        )
        self._warm = False

    async def warm_initial_toolset(self) -> None:
        """Warm the supplied initial tool set before the service becomes ready."""
        if self._active is None:
            return
        async with self._lock:
            await self._warm_candidate(self._active)
            self._warm = True

    async def update_toolset(self, records: Sequence[Mapping[str, Any]]) -> ToolConfig:
        """Warm and atomically publish a new tool set, retaining the old set on failure."""
        candidate = ToolConfig.from_records(
            records,
            model_config=self.model_config,
            max_tools=self.settings.max_tools,
        )
        async with self._lock:
            await self._warm_candidate(candidate)
            self._active = candidate
            self._warm = True
            return candidate

    async def classify(self, query: str) -> ClassificationResult:
        """Classify against an immutable active snapshot under the inference lock."""
        async with self._lock:
            if self._active is None or not self._warm:
                raise GatewayNotReady("No warmed tool set is active")
            prompt = render_full_prompt(
                self.tokenizer, query, self._active.tool_set, self.model_config
            )
            raw_output = await self.llama_server.completion(
                prompt=prompt,
                grammar=self._active.gbnf_grammar,
                n_predict=1,
                cache_prompt=True,
            )
            return classify_completion(raw_output, self._active.tool_set, self.model_config)

    async def ready(self) -> bool:
        try:
            healthy = await self.llama_server.health()
        except LlamaServerError:
            return False
        return healthy and self._active is not None and self._warm

    async def status(self) -> ToolSetStatusResponse:
        async with self._lock:
            if self._active is None:
                return ToolSetStatusResponse(toolset_version=None, tool_count=0, warm=False)
            return ToolSetStatusResponse(
                toolset_version=self._active.version,
                tool_count=len(self._active.tool_set.tools),
                warm=self._warm,
            )

    async def _warm_candidate(self, candidate: ToolConfig) -> None:
        prompt = render_warm_prompt(self.tokenizer, candidate.tool_set, self.model_config)
        await self.llama_server.completion(
            prompt=prompt,
            grammar=candidate.gbnf_grammar,
            n_predict=0,
            cache_prompt=True,
        )


def create_app(
    *,
    model_config: ServiceModelConfig,
    tokenizer: Any,
    llama_server: LlamaServerProtocol,
    settings: GatewaySettings,
    initial_tools: Sequence[Mapping[str, Any]] | None = None,
) -> FastAPI:
    """Create the gateway with injected runtime dependencies."""
    state = GatewayState(
        model_config=model_config,
        tokenizer=tokenizer,
        llama_server=llama_server,
        settings=settings,
        initial_tools=initial_tools,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await state.warm_initial_toolset()
        try:
            yield
        finally:
            await llama_server.close()

    app = FastAPI(title="Edge Tool Router", version="0.1.0", lifespan=lifespan)

    async def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
        if x_admin_token is None or not secrets.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Administrative credential required",
            )

    @app.post("/v1/classify", response_model=ClassificationResponse)
    async def classify(request: ClassifyRequest) -> ClassificationResponse:
        try:
            result = await state.classify(request.query)
        except GatewayNotReady as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
        except LlamaServerError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference engine unavailable",
            ) from exc
        return ClassificationResponse(**result.public_payload())

    @app.put(
        "/v1/toolset", response_model=ToolSetStatusResponse, dependencies=[Depends(require_admin)]
    )
    async def replace_toolset(tools: list[ToolInput]) -> ToolSetStatusResponse:
        try:
            await state.update_toolset([tool.model_dump() for tool in tools])
        except ToolSetValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
            ) from exc
        except LlamaServerError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference engine unavailable; previous tool set remains active",
            ) from exc
        return await state.status()

    @app.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready(response: Response) -> dict[str, str]:
        if await state.ready():
            return {"status": "ready"}
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "not_ready"}

    @app.get(
        "/v1/toolset/status",
        response_model=ToolSetStatusResponse,
        dependencies=[Depends(require_admin)],
    )
    async def toolset_status() -> ToolSetStatusResponse:
        return await state.status()

    app.state.gateway = state
    return app
