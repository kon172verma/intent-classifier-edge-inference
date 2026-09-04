"""Validated, release-compatible domain layer for edge tool routing."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation_lib.compatibility import PromptSpec, parse_prediction, positional_id_for_index
from evaluation_lib.prompt import build_full_prompt, build_user_message

MAX_POSITIONAL_TOOLS = 52
_POSITIONAL_IDS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ServiceConfigurationError(ValueError):
    """Raised when the pinned service configuration is invalid."""


class ToolSetValidationError(ValueError):
    """Raised when an administrative tool-set submission is unsafe or invalid."""


def _normalized_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _has_control_character(value: str) -> bool:
    return any(unicodedata.category(character).startswith("C") for character in value)


@dataclass(frozen=True)
class Tool:
    """One ordered tool definition supplied to the service."""

    name: str
    description: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Tool:
        name = value.get("name")
        description = value.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            raise ToolSetValidationError("Each tool requires string name and description fields")
        name = _normalized_text(name)
        description = _normalized_text(description)
        if not name or not description:
            raise ToolSetValidationError("Tool name and description must not be empty")
        if _has_control_character(name):
            raise ToolSetValidationError("Tool names must not contain control characters or newlines")
        if _has_control_character(description):
            raise ToolSetValidationError("Tool descriptions must not contain control characters")
        return cls(name=name, description=description)

    def as_prompt_tool(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True)
class ToolSet:
    """An ordered, validated tool set with deterministic content versioning."""

    tools: tuple[Tool, ...]
    manifest_version: str
    prompt_template_id: str

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        manifest_version: str,
        prompt_template_id: str,
        max_tools: int = MAX_POSITIONAL_TOOLS,
    ) -> ToolSet:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise ToolSetValidationError("Tool set must be an ordered list of tool objects")
        if not records:
            raise ToolSetValidationError("Tool set must contain at least one tool")
        if not 1 <= max_tools <= MAX_POSITIONAL_TOOLS:
            raise ToolSetValidationError(f"max_tools must be between 1 and {MAX_POSITIONAL_TOOLS}")
        if len(records) > max_tools:
            raise ToolSetValidationError(f"Tool set exceeds configured maximum of {max_tools} tools")
        tools = tuple(Tool.from_mapping(record) for record in records)
        names = [tool.name for tool in tools]
        if len(set(names)) != len(names):
            raise ToolSetValidationError("Tool names must be unique after normalization")
        return cls(tools, manifest_version, prompt_template_id)

    @property
    def version(self) -> str:
        payload = {
            "manifest_version": self.manifest_version,
            "prompt_template_id": self.prompt_template_id,
            "tools": [tool.as_prompt_tool() for tool in self.tools],
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def prompt_tools(self) -> list[dict[str, str]]:
        return [tool.as_prompt_tool() for tool in self.tools]

    def tool_for_id(self, tool_id: str) -> str | None:
        if tool_id == "-":
            return None
        try:
            index = _POSITIONAL_IDS.index(tool_id)
        except ValueError:
            return None
        return self.tools[index].name if index < len(self.tools) else None

    def gbnf_grammar(self) -> str:
        literals = [f'"{positional_id_for_index(index)}"' for index in range(len(self.tools))]
        literals.append('"-"')
        return f"root ::= {' | '.join(literals)}\n"


@dataclass(frozen=True)
class ServiceModelConfig:
    """The fixed v2.1 model and prompt contract declared by the service matrix."""

    manifest_version: str
    model_name: str
    quantization: str
    artifact_path: str
    tokenizer_path: str
    context_limit: int
    prompt_spec: PromptSpec

    @classmethod
    def load(cls, matrix_path: Path, manifest_path: Path) -> ServiceModelConfig:
        matrix = _read_json_object(matrix_path, "model selection matrix")
        manifest = _read_json_object(manifest_path, "manifest")
        model = matrix.get("model")
        prediction = matrix.get("prediction")
        if not isinstance(model, Mapping) or not isinstance(prediction, Mapping):
            raise ServiceConfigurationError("Service matrix requires model and prediction objects")

        version = _required_string(model, "manifest_version", "service matrix model")
        if manifest.get("version") != version:
            raise ServiceConfigurationError("Service matrix manifest version does not match the manifest")
        prompt = manifest.get("prompt")
        if not isinstance(prompt, dict):
            raise ServiceConfigurationError("Manifest prompt must be an object")
        prompt_spec = PromptSpec.from_manifest(prompt)
        if prompt_spec.output_format != "positional_id" or prompt_spec.template_id != "v2-positional-id":
            raise ServiceConfigurationError("Service requires the v2 positional-ID prompt contract")
        if prediction.get("model_output_format") != prompt_spec.output_format:
            raise ServiceConfigurationError("Service matrix output format does not match the manifest")
        if prediction.get("model_no_tool_token") != prompt_spec.model_no_tool_token:
            raise ServiceConfigurationError("Service matrix no-tool token does not match the manifest")
        if prediction.get("qwen_enable_thinking") is not False:
            raise ServiceConfigurationError("Qwen thinking must be disabled for service inference")

        model_name = _required_string(model, "name", "service matrix model")
        manifest_model = next(
            (
                item
                for item in manifest.get("models", [])
                if isinstance(item, Mapping) and item.get("name") == model_name
            ),
            None,
        )
        if manifest_model is None:
            raise ServiceConfigurationError("Service model is absent from the pinned manifest")
        release = model.get("release")
        manifest_release = manifest.get("release")
        if not isinstance(release, Mapping) or not isinstance(manifest_release, Mapping):
            raise ServiceConfigurationError("Service matrix and manifest require pinned release metadata")
        for key in ("repository", "revision"):
            if release.get(key) != manifest_release.get(key):
                raise ServiceConfigurationError(f"Service release {key} does not match the manifest")
        if release.get("subfolder") != manifest_model.get("release_subfolder"):
            raise ServiceConfigurationError("Service release subfolder does not match the manifest model")

        return cls(
            manifest_version=version,
            model_name=model_name,
            quantization=_required_string(model, "quantization", "service matrix model"),
            artifact_path=_required_string(model, "artifact_path", "service matrix model"),
            tokenizer_path=_required_string(model, "tokenizer_path", "service matrix model"),
            context_limit=_required_positive_int(model, "context_limit", "service matrix model"),
            prompt_spec=prompt_spec,
        )


@dataclass(frozen=True)
class ToolConfig:
    """Immutable active routing configuration for one validated tool set."""

    tool_set: ToolSet
    model_config: ServiceModelConfig

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, Any]],
        *,
        model_config: ServiceModelConfig,
        max_tools: int = MAX_POSITIONAL_TOOLS,
    ) -> ToolConfig:
        return cls(
            tool_set=ToolSet.from_records(
                records,
                manifest_version=model_config.manifest_version,
                prompt_template_id=model_config.prompt_spec.template_id,
                max_tools=max_tools,
            ),
            model_config=model_config,
        )

    @property
    def version(self) -> str:
        return self.tool_set.version

    @property
    def gbnf_grammar(self) -> str:
        return self.tool_set.gbnf_grammar()


@dataclass(frozen=True)
class ClassificationResult:
    """Safe public classification plus internal raw-completion provenance."""

    tool_id: str
    tool: str
    toolset_version: str
    raw_model_output: str
    invalid_output_reason: str | None

    def public_payload(self) -> dict[str, str]:
        return {
            "tool_id": self.tool_id,
            "tool": self.tool,
            "toolset_version": self.toolset_version,
        }


def render_user_prompt(query: str, tool_set: ToolSet, config: ServiceModelConfig) -> str:
    """Render the exact v2 user message without requiring a tokenizer."""
    _validate_query(query)
    return build_user_message(query, tool_set.prompt_tools(), config.prompt_spec)


def render_full_prompt(
    tokenizer: Any, query: str, tool_set: ToolSet, config: ServiceModelConfig
) -> str:
    """Render the exact tokenizer chat-template prompt used by the selected release."""
    _validate_query(query)
    return build_full_prompt(tokenizer, query, tool_set.prompt_tools(), config.prompt_spec)


def render_warm_prompt(tokenizer: Any, tool_set: ToolSet, config: ServiceModelConfig) -> str:
    """Render the exact empty-query prompt used to prefill a tool-set prefix."""
    return build_full_prompt(tokenizer, "", tool_set.prompt_tools(), config.prompt_spec)


def classify_completion(
    raw_model_output: str, tool_set: ToolSet, config: ServiceModelConfig
) -> ClassificationResult:
    """Map a completion to a valid public tool response."""
    if not isinstance(raw_model_output, str):
        raise TypeError("raw_model_output must be a string")
    parsed = parse_prediction(raw_model_output, tool_set.prompt_tools(), config.prompt_spec)
    if parsed.invalid_reason is not None:
        return ClassificationResult(
            tool_id="-",
            tool=config.prompt_spec.canonical_no_tool_value,
            toolset_version=tool_set.version,
            raw_model_output=raw_model_output,
            invalid_output_reason=parsed.invalid_reason,
        )
    if parsed.canonical_tool_name == config.prompt_spec.canonical_no_tool_value:
        return ClassificationResult(
            tool_id="-",
            tool=config.prompt_spec.canonical_no_tool_value,
            toolset_version=tool_set.version,
            raw_model_output=raw_model_output,
            invalid_output_reason=None,
        )
    assert parsed.canonical_tool_name is not None
    return ClassificationResult(
        tool_id=parsed.parsed_output,
        tool=parsed.canonical_tool_name,
        toolset_version=tool_set.version,
        raw_model_output=raw_model_output,
        invalid_output_reason=None,
    )


def _validate_query(query: str) -> None:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not query.strip():
        raise ToolSetValidationError("query must not be empty")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ServiceConfigurationError(f"Unable to read {label}: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ServiceConfigurationError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ServiceConfigurationError(f"{label} must be a JSON object")
    return value


def _required_string(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ServiceConfigurationError(f"{label}.{key} must be a non-empty string")
    return result


def _required_positive_int(value: Mapping[str, Any], key: str, label: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool) or result <= 0:
        raise ServiceConfigurationError(f"{label}.{key} must be a positive integer")
    return result
