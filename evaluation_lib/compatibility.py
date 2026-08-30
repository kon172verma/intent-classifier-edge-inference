"""Manifest-driven prompt and output compatibility for intent-classifier releases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

OutputFormat = Literal["tool_name", "positional_id"]
_POSITIONAL_IDS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class PromptSpec:
    """Release-specific prompt and output contract declared by a manifest."""

    system_prompt: str
    output_format: OutputFormat
    model_no_tool_token: str
    canonical_no_tool_value: str
    template_id: str

    @classmethod
    def from_manifest(cls, prompt: dict[str, Any]) -> PromptSpec:
        output_format = prompt["output_format"]
        if output_format not in {"tool_name", "positional_id"}:
            raise ValueError(f"Unsupported output format: {output_format!r}")
        return cls(
            system_prompt=str(prompt["system_prompt"]),
            output_format=output_format,
            model_no_tool_token=str(prompt["model_no_tool_token"]),
            canonical_no_tool_value=str(prompt["canonical_no_tool_value"]),
            template_id=str(prompt["template_id"]),
        )


@dataclass(frozen=True)
class ParsedPrediction:
    """Raw and canonical representations of one model generation."""

    raw_output: str
    parsed_output: str
    canonical_tool_name: str | None
    invalid_reason: str | None


def legacy_prompt_spec(system_prompt: str) -> PromptSpec:
    """Return the direct-command contract used before version manifests existed."""
    return PromptSpec(
        system_prompt=system_prompt,
        output_format="tool_name",
        model_no_tool_token="none",
        canonical_no_tool_value="none",
        template_id="legacy-tool-name",
    )


def canonical_expected(answer: str, spec: PromptSpec) -> str:
    """Convert the dataset's no-tool spelling into the shared canonical value."""
    if answer in {spec.model_no_tool_token, spec.canonical_no_tool_value, "none", "-"}:
        return spec.canonical_no_tool_value
    return answer


def positional_id_for_index(index: int) -> str:
    """Return the manifest positional ID for a zero-based tool-list index."""
    if not 0 <= index < len(_POSITIONAL_IDS):
        raise ValueError(f"Tool list has no supported positional ID at index {index}")
    return _POSITIONAL_IDS[index]


def _strip_reasoning(text: str) -> str:
    if "<think>" in text and "</think>" in text:
        return text.split("</think>", 1)[-1]
    return text


def _first_output_token(text: str) -> str:
    return _strip_reasoning(text).strip().split("\n", 1)[0].strip().rstrip(".,;:")


def parse_prediction(
    raw_output: str,
    available_tools: list[dict[str, Any]],
    spec: PromptSpec,
) -> ParsedPrediction:
    """Parse native model output and return its canonical readable tool name.

    Positional IDs are decoded against the tool order of this individual
    example. A static mapping would be incorrect because each prompt contains
    a different ordered tool subset.
    """
    parsed = _first_output_token(raw_output)
    names = [str(tool["name"]) for tool in available_tools]
    no_tool_tokens = {
        spec.model_no_tool_token,
        spec.canonical_no_tool_value,
        "none",
        "-",
    }
    if parsed in no_tool_tokens:
        return ParsedPrediction(raw_output, parsed, spec.canonical_no_tool_value, None)

    if spec.output_format == "positional_id":
        if len(parsed) != 1 or parsed not in _POSITIONAL_IDS:
            return ParsedPrediction(raw_output, parsed, None, "expected_positional_id")
        index = _POSITIONAL_IDS.index(parsed)
        if index >= len(names):
            return ParsedPrediction(raw_output, parsed, None, "positional_id_out_of_range")
        return ParsedPrediction(raw_output, parsed, names[index], None)

    if parsed in names:
        return ParsedPrediction(raw_output, parsed, parsed, None)
    for name in sorted(names, key=len, reverse=True):
        if name in _strip_reasoning(raw_output):
            return ParsedPrediction(raw_output, parsed, name, None)
    return ParsedPrediction(raw_output, parsed, None, "unknown_tool_name")
