"""Tests for version-specific prompt output compatibility."""

from __future__ import annotations

import unittest

from evaluation_lib.compatibility import PromptSpec, canonical_expected, parse_prediction


class OutputCompatibilityTests(unittest.TestCase):
    """Ensure quality is measured on canonical readable tool names."""

    def setUp(self) -> None:
        self.tools = [{"name": "call_handler"}, {"name": "nav_route_planner"}]
        self.v1 = PromptSpec("route", "tool_name", "none", "none", "v1-tool-name")
        self.v2 = PromptSpec("route", "positional_id", "-", "none", "v2-positional-id")

    def test_tool_name_output_preserves_the_readable_name(self) -> None:
        parsed = parse_prediction("<think>ignore</think>\ncall_handler", self.tools, self.v1)
        self.assertEqual(parsed.parsed_output, "call_handler")
        self.assertEqual(parsed.canonical_tool_name, "call_handler")
        self.assertIsNone(parsed.invalid_reason)

    def test_positional_id_decodes_against_this_examples_tool_order(self) -> None:
        parsed = parse_prediction("b", self.tools, self.v2)
        self.assertEqual(parsed.parsed_output, "b")
        self.assertEqual(parsed.canonical_tool_name, "nav_route_planner")
        self.assertIsNone(parsed.invalid_reason)

    def test_no_tool_and_invalid_positional_output_are_explicit(self) -> None:
        no_tool = parse_prediction("-", self.tools, self.v2)
        invalid = parse_prediction("z", self.tools, self.v2)
        self.assertEqual(no_tool.canonical_tool_name, "none")
        self.assertEqual(canonical_expected("-", self.v2), "none")
        self.assertIsNone(invalid.canonical_tool_name)
        self.assertEqual(invalid.invalid_reason, "positional_id_out_of_range")


if __name__ == "__main__":
    unittest.main()
