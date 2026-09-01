"""Tests for generation-configuration compatibility helpers."""

from __future__ import annotations

import unittest

from evaluation_lib.generation import normalize_token_ids


class GenerationCompatibilityTests(unittest.TestCase):
    """EOS token IDs may be scalar or collection-valued across model families."""

    def test_normalizes_scalar_eos_token_id(self) -> None:
        self.assertEqual(normalize_token_ids(2), {2})

    def test_normalizes_multiple_eos_token_ids(self) -> None:
        self.assertEqual(normalize_token_ids([2, 151643]), {2, 151643})

    def test_normalizes_missing_eos_token_id(self) -> None:
        self.assertEqual(normalize_token_ids(None), set())


if __name__ == "__main__":
    unittest.main()
