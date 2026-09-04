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

    At least the final full-prompt token remains in the query segment. Some
    chat templates render an empty user message with extra delimiter tokens;
    in that case the empty-request token sequence can contain the entire full
    prompt as a prefix. Keeping the final token out of the prefilled cache
    gives generation a non-empty input sequence without duplicating a token.
    """
    n = min(len(full_tokens), len(tools_only_tokens))
    for i in range(n):
        if full_tokens[i] != tools_only_tokens[i]:
            return i
    if not full_tokens:
        return 0
    return min(n, len(full_tokens) - 1)
