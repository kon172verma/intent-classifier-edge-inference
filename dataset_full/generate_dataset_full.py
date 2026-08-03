#!/usr/bin/env python3
"""
Full 10,000-example dataset generator for the tool_router project.

Generates 100 JSON files, each with exactly 100 examples:
  sample_0001.json  …  sample_0100.json

File 001 is byte-for-byte identical to dataset_sample/sample.json.
All 10,000 examples are globally unique (deduplicated on the triple:
    user_request × frozenset(available_tool_names) × answer).

Distribution targets (per project spec):
  * 10% of examples: 1–3 tools  (few-tool regime)
  * 80% of examples: 4–19 tools (standard regime)
  * 10% of examples: 20–30 tools (many-tool regime)
  * ~20% none answers, ~80% valid-tool answers
  * Rare tools ≤ 2–3% of examples as correct answer

Usage:
    python generate_dataset_full.py                     # writes files next to this script
    python generate_dataset_full.py --seed 42 --out-dir /some/path
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Import generation helpers from the sibling dataset_sample package
# ---------------------------------------------------------------------------
_SAMPLE_DIR = Path(__file__).parent.parent / "dataset_sample"
sys.path.insert(0, str(_SAMPLE_DIR))

from generate_dataset import (  # type: ignore[import]
    build_example,
    load_tools,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOOLS_REF: Path = _SAMPLE_DIR / "tools_reference.json"
SAMPLE_JSON: Path = _SAMPLE_DIR / "sample.json"

FILES: int = 100
EXAMPLES_PER_FILE: int = 100
TOTAL: int = FILES * EXAMPLES_PER_FILE  # 10,000

RARE_TOOL_NAMES: frozenset[str] = frozenset({
    "emergency_sos",
    "roadside_assistance",
    "insurance_claims",
    "home_automation_bridge",
    "corp-fleet-manager",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fingerprint(example: dict) -> tuple:
    """Canonical hashable identity for an example."""
    return (
        example["user_request"],
        frozenset(t["name"] for t in example["available_tools"]),
        example["answer"],
    )


def _generate_with_distribution(
    n: int,
    all_tools: list[dict],
    common_tools: list[dict],
    rare_tools: list[dict],
    seen: set[tuple],
    rng: random.Random,
) -> list[dict]:
    """
    Generate up to `n` unique examples (not already in `seen`),
    following the project distribution spec.
    Returns a list that may be slightly shorter than `n` if the rejection
    sampler exhausts its budget (extremely unlikely given the sample space).
    """
    n_none = round(n * 0.20)
    n_valid = n - n_none
    n_rare = max(1, round(n * 0.02))
    n_common = n_valid - n_rare

    n_few = round(n * 0.10)    # 1–3 tools
    n_many = round(n * 0.10)   # 20–30 tools
    n_std = n - n_few - n_many  # 4–19 tools

    # Build answer pool
    rare_pool: list[dict | None] = [rng.choice(rare_tools) for _ in range(n_rare)]
    common_pool: list[dict | None] = rng.choices(common_tools, k=n_common)
    none_pool: list[dict | None] = [None] * n_none
    answer_pool = rare_pool + common_pool + none_pool
    rng.shuffle(answer_pool)

    # Build tool-count bucket list
    max_tools = len(all_tools)
    count_buckets: list[int] = (
        [rng.randint(1, 3) for _ in range(n_few)]
        + [rng.randint(20, min(30, max_tools)) for _ in range(n_many)]
        + [rng.randint(4, 19) for _ in range(n_std)]
    )
    rng.shuffle(count_buckets)

    results: list[dict] = []

    for tool, base_count in zip(answer_pool, count_buckets):
        base_count = max(1, min(base_count, max_tools))
        accepted = False

        # Primary attempts: stay near the intended bucket count
        for attempt in range(60):
            jitter = rng.randint(-2, 2) if attempt > 0 else 0
            count = max(1, min(max_tools, base_count + jitter))
            ex = build_example(all_tools, tool, count)
            fp = fingerprint(ex)
            if fp not in seen:
                seen.add(fp)
                results.append(ex)
                accepted = True
                break

        if not accepted:
            # Extended fallback: random count, higher retry budget
            for _ in range(300):
                count = rng.randint(1, max_tools)
                ex = build_example(all_tools, tool, count)
                fp = fingerprint(ex)
                if fp not in seen:
                    seen.add(fp)
                    results.append(ex)
                    accepted = True
                    break

        # If still not accepted after extended retries, skip silently.
        # With 265 distinct request strings and billions of (request, tool_subset)
        # combinations, this branch should never be reached.

    return results


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

def generate_full(seed: int = 42) -> list[list[dict]]:
    """
    Returns 100 lists of 100 examples each.

    File 0 (index 0 of the returned list) is loaded directly from
    dataset_sample/sample.json so it is byte-for-byte identical.
    Files 1–99 are freshly generated with guaranteed global uniqueness.
    """
    # ── File 1: load unchanged from sample.json ──────────────────────────
    file1_examples: list[dict] = json.loads(
        SAMPLE_JSON.read_text(encoding="utf-8")
    )
    if len(file1_examples) != EXAMPLES_PER_FILE:
        raise ValueError(
            f"dataset_sample/sample.json contains {len(file1_examples)} examples; "
            f"expected {EXAMPLES_PER_FILE}."
        )

    seen: set[tuple] = {fingerprint(e) for e in file1_examples}

    # ── Files 2–100: generate fresh unique examples ───────────────────────
    common_tools, rare_tools = load_tools(TOOLS_REF)
    all_tools = common_tools + rare_tools

    # Use a separate RNG instance seeded independently from sample.json (seed=42)
    rng = random.Random(seed + 1)  # seed=43 by default

    # Temporarily swap the module-level random state so build_example (which
    # calls the top-level random.* functions) uses our controlled RNG.
    _old_state = random.getstate()
    random.setstate(rng.getstate())

    n_remaining = TOTAL - EXAMPLES_PER_FILE  # 9,900
    print(f"  Generating {n_remaining} additional unique examples...")
    remaining = _generate_with_distribution(
        n_remaining, all_tools, common_tools, rare_tools, seen, rng
    )

    # Restore original random state
    random.setstate(_old_state)

    if len(remaining) < n_remaining:
        print(
            f"  WARNING: generated only {len(remaining)} unique examples "
            f"(target {n_remaining}). Increase retry budget if needed."
        )

    # ── Assemble 100 files ────────────────────────────────────────────────
    all_examples: list[dict] = file1_examples + remaining
    files: list[list[dict]] = [
        all_examples[i * EXAMPLES_PER_FILE : (i + 1) * EXAMPLES_PER_FILE]
        for i in range(FILES)
    ]
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the full 10,000-example tool_router dataset "
            "(100 files × 100 examples each)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed. File 001 always equals sample.json (seed=42); "
             "subsequent files use seed+1.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path(__file__).parent,
        help="Directory to write sample_NNNN.json files.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating full dataset ({TOTAL:,} examples, {FILES} files × {EXAMPLES_PER_FILE}) …")
    files = generate_full(seed=args.seed)

    for idx, examples in enumerate(files, start=1):
        fname = args.out_dir / f"sample_{idx:04d}.json"
        fname.write_text(
            json.dumps(examples, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"\nWrote {len(files)} files → {args.out_dir}")

    # ── Distribution summary ─────────────────────────────────────────────
    all_examples = [ex for batch in files for ex in batch]
    answers = [e["answer"] for e in all_examples]
    counts = [len(e["available_tools"]) for e in all_examples]
    n = len(all_examples)

    n_none = answers.count("none")
    n_rare = sum(1 for a in answers if a in RARE_TOOL_NAMES)
    n_few = sum(1 for c in counts if c <= 3)
    n_std = sum(1 for c in counts if 4 <= c <= 19)
    n_many = sum(1 for c in counts if c >= 20)

    fps = {fingerprint(e) for e in all_examples}

    print(f"\nDistribution summary (n={n:,}):")
    print(f"  none answers       : {n_none:5d}  ({n_none / n:.1%})")
    print(f"  rare-tool correct  : {n_rare:5d}  ({n_rare / n:.1%})")
    print(f"  few-tool  (1–3)    : {n_few:5d}  ({n_few / n:.1%})")
    print(f"  standard  (4–19)   : {n_std:5d}  ({n_std / n:.1%})")
    print(f"  many-tool (20–30)  : {n_many:5d}  ({n_many / n:.1%})")
    print(f"  unique examples    : {len(fps):5d} / {n:,}")


if __name__ == "__main__":
    main()
