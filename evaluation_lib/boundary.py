"""Generic token-sequence boundary helper shared by all inference backends."""

from __future__ import annotations

from collections.abc import Sequence


def find_tools_query_boundary(
    full_tokens: Sequence[int],
    tools_only_tokens: Sequence[int],
) -> int:
    """Return the index within *full_tokens* where the user-query tokens begin.

    Computed as the length of the common prefix between the full prompt tokens
    and an equivalent prompt built with an empty user request (see
    ``build_tools_only_prompt``). Robust to tokenizer merge effects.
    """
    n = min(len(full_tokens), len(tools_only_tokens))
    for i in range(n):
        if full_tokens[i] != tools_only_tokens[i]:
            return i
    return n
