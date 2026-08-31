"""Regression tests for release-specific prompt rendering."""

from __future__ import annotations

import unittest
from typing import Any

from evaluation_lib.compatibility import PromptSpec
from evaluation_lib.prompt import build_full_prompt, build_user_message


class RecordingTokenizer:
    """Minimal tokenizer that exposes the messages passed to its chat template."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.kwargs: dict[str, Any] = {}

    def apply_chat_template(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.messages = messages
        self.kwargs = kwargs
        return "rendered prompt"


class PromptTemplateTests(unittest.TestCase):
    """Keep v1's training prompt contract separate from v2's ID contract."""

    def setUp(self) -> None:
        self.tools = [
            {"name": "nav_route_planner", "description": "Plans a driving route."},
            {"name": "climate_control", "description": "Adjusts cabin temperature."},
        ]
        self.v1 = PromptSpec("You are a tool router.", "tool_name", "none", "none", "v1-tool-name")
        self.v2 = PromptSpec(
            "You are a tool router.", "positional_id", "-", "none", "v2-positional-id"
        )

    def test_v1_user_message_matches_the_fine_tuning_layout(self) -> None:
        self.assertEqual(
            build_user_message("Take me downtown.", self.tools, self.v1),
            "Available Tools:\n"
            "Name: nav_route_planner\n"
            "Description: Plans a driving route.\n\n"
            "Name: climate_control\n"
            "Description: Adjusts cabin temperature.\n\n"
            "User Request:\n"
            "Take me downtown.\n\n"
            "Selected Tool:",
        )

    def test_v1_full_prompt_uses_the_versioned_user_message(self) -> None:
        tokenizer = RecordingTokenizer()
        self.assertEqual(
            build_full_prompt(tokenizer, "Take me downtown.", self.tools, self.v1),
            "rendered prompt",
        )
        self.assertEqual(
            tokenizer.messages[0], {"role": "system", "content": self.v1.system_prompt}
        )
        self.assertEqual(
            tokenizer.messages[1]["content"],
            build_user_message("Take me downtown.", self.tools, self.v1),
        )
        self.assertTrue(tokenizer.kwargs["add_generation_prompt"])

    def test_v2_user_message_matches_the_positional_id_fine_tuning_layout(self) -> None:
        self.assertEqual(
            build_user_message("Take me downtown.", self.tools, self.v2),
            "Available Tools:\n"
            "ID | Name | Description\n"
            "a | nav_route_planner | Plans a driving route.\n"
            "b | climate_control | Adjusts cabin temperature.\n\n"
            "User Request:\n"
            "Take me downtown.\n\n"
            "Selected Tool:",
        )


if __name__ == "__main__":
    unittest.main()
