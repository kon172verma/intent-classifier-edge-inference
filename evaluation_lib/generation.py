"""Small compatibility helpers for generation configuration values."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_token_ids(token_ids: int | Iterable[int] | None) -> set[int]:
    """Return generation token IDs as a set.

    Hugging Face configurations legitimately represent ``eos_token_id`` as
    either one integer (for example SmolLM2) or a collection of integers.
    Evaluators use a set for efficient stop-token membership checks.
    """
    if token_ids is None:
        return set()
    if isinstance(token_ids, int):
        return {token_ids}
    if isinstance(token_ids, (str, bytes)):
        raise TypeError("Token IDs must be an integer or an iterable of integers")
    return {int(token_id) for token_id in token_ids}
