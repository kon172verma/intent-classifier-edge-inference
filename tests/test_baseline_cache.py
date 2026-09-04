"""Regression tests for Transformers prefix-cache compatibility."""

import torch

from evaluation_baseline.cache import clone_cache, kv_cache_tokens


def test_clone_legacy_cache_is_independent_and_preserves_length() -> None:
    """Legacy tuple caches remain valid on both old and new Transformers APIs."""
    key = torch.ones((1, 2, 3, 4))
    value = torch.ones((1, 2, 3, 4))
    cloned = clone_cache(((key, value),))

    assert cloned is not None
    assert kv_cache_tokens(cloned) == 3
    assert cloned[0][0] is not key
    assert cloned[0][1] is not value
