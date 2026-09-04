"""Unit tests for the edge-service domain contract."""

from __future__ import annotations

import unittest
from pathlib import Path

from service_edge_inference.domain import (
    MAX_POSITIONAL_TOOLS,
    ServiceModelConfig,
    ToolConfig,
    ToolSet,
    ToolSetValidationError,
    classify_completion,
    render_user_prompt,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MATRIX_PATH = REPO_ROOT / "service_edge_inference" / "model_selection_matrix.json"
MANIFEST_PATH = REPO_ROOT / "manifests" / "v2.1.json"


class ServiceDomainTests(unittest.TestCase):
    """Keep the service's public contract compatible with v2.1 evaluation."""

    def setUp(self) -> None:
        self.config = ServiceModelConfig.load(MATRIX_PATH, MANIFEST_PATH)
        self.records = [
            {"name": "call_handler", "description": "Call a registered handler."},
            {"name": "nav_route_planner", "description": "Plan a driving route."},
        ]
        self.tool_set = ToolSet.from_records(
            self.records,
            manifest_version=self.config.manifest_version,
            prompt_template_id=self.config.prompt_spec.template_id,
        )

    def test_fixed_service_configuration_matches_v21_manifest(self) -> None:
        self.assertEqual(self.config.model_name, "Qwen2.5-0.5B")
        self.assertEqual(self.config.quantization, "Q6_K")
        self.assertEqual(
            self.config.tokenizer_path,
            "models/v2.1/Qwen2.5-0.5B/transformers/merged",
        )
        self.assertEqual(self.config.prompt_spec.output_format, "positional_id")
        self.assertEqual(self.config.prompt_spec.model_no_tool_token, "-")

    def test_tool_order_controls_ids_and_hashing(self) -> None:
        reversed_tool_set = ToolSet.from_records(
            list(reversed(self.records)),
            manifest_version=self.config.manifest_version,
            prompt_template_id=self.config.prompt_spec.template_id,
        )
        self.assertEqual(self.tool_set.tool_for_id("a"), "call_handler")
        self.assertEqual(self.tool_set.tool_for_id("b"), "nav_route_planner")
        self.assertNotEqual(self.tool_set.version, reversed_tool_set.version)

    def test_tool_config_is_an_immutable_snapshot_of_model_and_toolset(self) -> None:
        config = ToolConfig.from_records(self.records, model_config=self.config)
        self.assertEqual(config.version, self.tool_set.version)
        self.assertEqual(config.gbnf_grammar, 'root ::= "a" | "b" | "-"\n')

    def test_validation_rejects_duplicates_empty_and_control_characters(self) -> None:
        cases = (
            (
                [
                    {"name": "route", "description": "One"},
                    {"name": "route", "description": "Two"},
                ],
                "unique",
            ),
            ([{"name": " ", "description": "Description"}], "must not be empty"),
            ([{"name": "route" + chr(10) + "name", "description": "Description"}], "control characters"),
        )
        for records, message in cases:
            with self.subTest(records=records):
                with self.assertRaisesRegex(ToolSetValidationError, message):
                    ToolSet.from_records(
                        records,
                        manifest_version=self.config.manifest_version,
                        prompt_template_id=self.config.prompt_spec.template_id,
                    )

    def test_validation_rejects_more_than_supported_positional_ids(self) -> None:
        records = [
            {"name": f"tool_{index}", "description": f"Description {index}"}
            for index in range(MAX_POSITIONAL_TOOLS + 1)
        ]
        with self.assertRaisesRegex(ToolSetValidationError, "maximum"):
            ToolSet.from_records(
                records,
                manifest_version=self.config.manifest_version,
                prompt_template_id=self.config.prompt_spec.template_id,
            )

    def test_prompt_and_grammar_preserve_exact_v2_id_contract(self) -> None:
        prompt = render_user_prompt("Take me downtown.", self.tool_set, self.config)
        self.assertEqual(
            prompt,
            "Available Tools:\n"
            "ID | Name | Description\n"
            "a | call_handler | Call a registered handler.\n"
            "b | nav_route_planner | Plan a driving route.\n\n"
            "User Request:\n"
            "Take me downtown.\n\n"
            "Selected Tool:",
        )
        self.assertEqual(self.tool_set.gbnf_grammar(), 'root ::= "a" | "b" | "-"\n')

    def test_valid_none_and_invalid_completions_are_safe_public_results(self) -> None:
        valid = classify_completion("b", self.tool_set, self.config)
        self.assertEqual(valid.public_payload()["tool_id"], "b")
        self.assertEqual(valid.public_payload()["tool"], "nav_route_planner")
        self.assertIsNone(valid.invalid_output_reason)

        no_tool = classify_completion("-", self.tool_set, self.config)
        self.assertEqual(no_tool.public_payload()["tool_id"], "-")
        self.assertEqual(no_tool.public_payload()["tool"], "none")

        invalid = classify_completion("z", self.tool_set, self.config)
        self.assertEqual(invalid.public_payload()["tool_id"], "-")
        self.assertEqual(invalid.public_payload()["tool"], "none")
        self.assertEqual(invalid.raw_model_output, "z")
        self.assertEqual(invalid.invalid_output_reason, "positional_id_out_of_range")


if __name__ == "__main__":
    unittest.main()
