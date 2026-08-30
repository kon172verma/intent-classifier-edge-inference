"""Backward-compatible wrapper around manifest-aware output parsing."""

from __future__ import annotations

from evaluation_lib.compatibility import PromptSpec, parse_prediction


def extract_predicted_tool(text: str, available_tool_names: set[str]) -> str:
    """Recover the predicted tool name from raw model output.

    Strategy:
    1. Strip any ``<think>...</think>`` block (Qwen3 thinking mode).
    2. Try an exact match on the first output line.
    3. Fall back to a substring search over all available tool names.
    4. Check for the literal word "none".
    5. Return the first line as-is (will be counted as invalid).
    """
    spec = PromptSpec(
        system_prompt="",
        output_format="tool_name",
        model_no_tool_token="none",
        canonical_no_tool_value="none",
        template_id="legacy-tool-name",
    )
    parsed = parse_prediction(text, [{"name": name} for name in available_tool_names], spec)
    return parsed.canonical_tool_name or parsed.parsed_output
