"""Prompt construction for tool-routing evaluation."""

from __future__ import annotations

from typing import Any

from evaluation_lib.compatibility import PromptSpec, legacy_prompt_spec
from evaluation_lib.config import SYSTEM_PROMPT

_DEFAULT_PROMPT_SPEC = legacy_prompt_spec(SYSTEM_PROMPT)


def build_user_message(user_request: str, available_tools: list[dict]) -> str:
    """Format the tool list and user request into a single user turn."""
    tool_lines = "\n".join(f"- {t['name']}: {t['description']}" for t in available_tools)
    return f"Available tools:\n{tool_lines}\n\nUser request: {user_request}"


def _apply_template(tokenizer: Any, messages: list[dict], **extra: Any) -> str:
    """Apply the tokenizer's chat template, disabling Qwen3 thinking if supported."""
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        **extra,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_full_prompt(
    tokenizer: Any,
    user_request: str,
    available_tools: list[dict],
    prompt_spec: PromptSpec | None = None,
) -> str:
    """Build the complete chat-template prompt for one inference example."""
    prompt_spec = prompt_spec or _DEFAULT_PROMPT_SPEC
    messages = [
        {"role": "system", "content": prompt_spec.system_prompt},
        {"role": "user", "content": build_user_message(user_request, available_tools)},
    ]
    return _apply_template(tokenizer, messages)


def build_system_prefix_text(tokenizer: Any, prompt_spec: PromptSpec | None = None) -> str:
    """Return only the formatted system turn (the cacheable static prefix).

    Returns an empty string if the tokenizer raises during template rendering.
    """
    prompt_spec = prompt_spec or _DEFAULT_PROMPT_SPEC
    messages = [{"role": "system", "content": prompt_spec.system_prompt}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        return ""


def build_tools_only_prompt(
    tokenizer: Any, available_tools: list[dict], prompt_spec: PromptSpec | None = None
) -> str:
    """Build the full chat-template prompt with an empty user request.

    Identical to :func:`build_full_prompt` except the user request is blank.
    Tokenising this alongside the real prompt lets callers find the token
    boundary between the (static, cacheable) tools-list prefix and the
    (dynamic) user-query suffix via a longest-common-prefix comparison,
    without having to parse or split the rendered chat-template string.
    """
    return build_full_prompt(tokenizer, "", available_tools, prompt_spec)
