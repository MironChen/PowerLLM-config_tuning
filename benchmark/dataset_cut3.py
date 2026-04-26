"""
Sample questions from LegalBench-RAG format datasets.

Reads all .json files under an input directory (default: benchmark/legalbench-datasets/),
groups tests by their source document (file_path in snippets), and draws a balanced
random sample: at least --min-per-context questions per context, across --target-contexts
distinct contexts, totalling --target-questions questions.

Output is a valid LegalBench-RAG file (uses the same "tests" key as the source) with
sampling metadata stored in sibling top-level fields.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_INPUT_PATH = Path(__file__).resolve().parent / "legalbench-datasets"
DEFAULT_TARGET_QUESTIONS = 200
DEFAULT_TARGET_CONTEXTS = 20
DEFAULT_MIN_PER_CONTEXT = 5
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load_file(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "tests" not in data:
        raise ValueError(
            f"{path}: expected a JSON object with a top-level 'tests' key."
        )

    tests = data["tests"]
    if not isinstance(tests, list):
        raise ValueError(f"{path}: 'tests' must be a list.")

    return tests


def load_all_tests(input_path: Path) -> list[dict[str, Any]]:
    """Load every test record from all .json files under input_path."""
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(p for p in input_path.rglob("*.json") if p.is_file())
        if not files:
            raise FileNotFoundError(f"No .json files found under {input_path}")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    all_tests: list[dict[str, Any]] = []
    for file_path in files:
        tests = _load_file(file_path)
        all_tests.extend(tests)

    if not all_tests:
        raise ValueError("No test records were loaded.")

    return all_tests


# ---------------------------------------------------------------------------
# Context grouping
# ---------------------------------------------------------------------------


def _primary_file_path(test: dict[str, Any]) -> str:
    """
    Return the file_path that appears most frequently across all snippets in a test.
    Ties are broken lexicographically. Raises if no file_path can be found.
    """
    snippets = test.get("snippets", [])
    counts: dict[str, int] = defaultdict(int)
    for snippet in snippets:
        fp = snippet.get("file_path", "").strip()
        if fp:
            counts[fp] += 1

    if not counts:
        raise ValueError(
            f"Test has no resolvable file_path in its snippets: {test.get('query', '')[:80]!r}"
        )

    # Sort by (-count, path) so the most-frequent path wins; lexicographic tiebreak.
    return min(counts, key=lambda p: (-counts[p], p))


def group_by_context(tests: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for test in tests:
        primary = _primary_file_path(test)
        groups[primary].append(test)
    return dict(groups)


def load_context_paths(dataset_path: Path) -> set[str]:
    """Load source document paths from a LegalBench-RAG dataset."""
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{dataset_path}: expected a JSON object.")

    paths: set[str] = set()
    contexts = data.get("contexts")
    if isinstance(contexts, list):
        for context in contexts:
            if not isinstance(context, dict):
                continue
            file_path = context.get("file_path")
            if isinstance(file_path, str) and file_path.strip():
                paths.add(file_path.strip())

    tests = data.get("tests")
    if isinstance(tests, list):
        for test in tests:
            if not isinstance(test, dict):
                continue
            snippets = test.get("snippets")
            if not isinstance(snippets, list):
                continue
            for snippet in snippets:
                if not isinstance(snippet, dict):
                    continue
                file_path = snippet.get("file_path")
                if isinstance(file_path, str) and file_path.strip():
                    paths.add(file_path.strip())

    if not paths:
        raise ValueError(
            f"{dataset_path}: no context file_path values found in contexts or snippets."
        )

    return paths


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_tests(
    *,
    tests: list[dict[str, Any]],
    target_questions: int,
    target_contexts: int,
    min_per_context: int,
    seed: int,
    exclude_contexts: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Return (sampled_tests, counts_by_context).

    Strategy:
      1. Group by primary file_path.
      2. Keep only contexts with >= min_per_context questions.
      3. Shuffle eligible contexts, then sort descending by question count so
         the selected pool can reliably reach the total target.
      4. Take the first target_contexts.
      5. Guarantee min_per_context questions from each; put the rest in a leftover pool.
      6. Fill remaining slots from the leftover pool via random.sample.
      7. Shuffle the final list.
    """
    if target_questions <= 0:
        raise ValueError("--target-questions must be a positive integer.")
    if target_contexts <= 0:
        raise ValueError("--target-contexts must be a positive integer.")
    if min_per_context <= 0:
        raise ValueError("--min-per-context must be a positive integer.")

    floor = target_contexts * min_per_context
    if target_questions < floor:
        raise ValueError(
            f"--target-questions ({target_questions}) must be >= "
            f"target_contexts * min_per_context ({floor})."
        )

    grouped = group_by_context(tests)
    if exclude_contexts:
        grouped = {
            fp: cases
            for fp, cases in grouped.items()
            if fp not in exclude_contexts
        }

    eligible = {
        fp: cases
        for fp, cases in grouped.items()
        if len(cases) >= min_per_context
    }

    if len(eligible) < target_contexts:
        raise ValueError(
            f"Need {target_contexts} contexts with at least {min_per_context} questions each, "
            f"but only {len(eligible)} eligible contexts exist."
        )

    rng = random.Random(seed)

    # Shuffle first for randomness, then stable-sort by count (descending) so
    # the top target_contexts have enough questions to cover the total target.
    shuffled = list(eligible.keys())
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda fp: len(eligible[fp]), reverse=True)
    selected_fps = shuffled[:target_contexts]

    total_available = sum(len(eligible[fp]) for fp in selected_fps)
    if total_available < target_questions:
        raise ValueError(
            f"The {target_contexts} selected contexts only have {total_available} questions "
            f"combined, but {target_questions} are required."
        )

    guaranteed: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for fp in selected_fps:
        pool = list(eligible[fp])
        rng.shuffle(pool)
        guaranteed.extend(pool[:min_per_context])
        leftovers.extend(pool[min_per_context:])
        counts[fp] = min_per_context

    remaining = target_questions - len(guaranteed)
    extra = rng.sample(leftovers, remaining)
    for test in extra:
        fp = _primary_file_path(test)
        counts[fp] = counts.get(fp, 0) + 1

    result = guaranteed + extra
    rng.shuffle(result)
    return result, counts


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def build_output(
    *,
    sampled: list[dict[str, Any]],
    counts: dict[str, int],
    input_path: Path,
    seed: int,
    target_questions: int,
    target_contexts: int,
    min_per_context: int,
    exclude_datasets: list[Path] | None = None,
    excluded_context_count: int = 0,
    excluded_matching_context_count: int = 0,
) -> dict[str, Any]:
    context_summary = [
        {"file_path": fp, "question_count": n}
        for fp, n in sorted(counts.items())
    ]

    return {
        "format": "legalbenchrag_sample_v2",
        "source_input": str(input_path),
        "sampling": {
            "seed": seed,
            "target_questions": target_questions,
            "target_contexts": target_contexts,
            "min_per_context": min_per_context,
            "questions_sampled": len(sampled),
            "contexts_sampled": len(counts),
            "exclude_datasets": (
                [str(path) for path in exclude_datasets]
                if exclude_datasets
                else []
            ),
            "excluded_context_count": excluded_context_count,
            "excluded_matching_context_count": excluded_matching_context_count,
        },
        "contexts": context_summary,
        "tests": sampled,
    }


def default_output_path(
    *,
    target_questions: int,
    target_contexts: int,
    min_per_context: int,
    seed: int,
) -> Path:
    name = (
        f"legalbench_sample_q{target_questions}_c{target_contexts}"
        f"_min{min_per_context}_seed{seed}.json"
    )
    return Path(__file__).resolve().parent / name


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Path to a LegalBench-RAG .json file or a directory of .json files. "
            f"Defaults to benchmark/legalbench-datasets/."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output file path. Defaults to a name derived from sampling parameters.",
    )
    parser.add_argument(
        "--target-questions",
        type=int,
        default=DEFAULT_TARGET_QUESTIONS,
        help=f"Total questions to sample (default: {DEFAULT_TARGET_QUESTIONS}).",
    )
    parser.add_argument(
        "--target-contexts",
        type=int,
        default=DEFAULT_TARGET_CONTEXTS,
        help=f"Number of distinct source documents to draw from (default: {DEFAULT_TARGET_CONTEXTS}).",
    )
    parser.add_argument(
        "--min-per-context",
        type=int,
        default=DEFAULT_MIN_PER_CONTEXT,
        help=f"Minimum questions guaranteed from each selected context (default: {DEFAULT_MIN_PER_CONTEXT}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--exclude-dataset",
        type=Path,
        action="append",
        default=[],
        metavar="FILE",
        help=(
            "LegalBench-RAG dataset whose contexts should be excluded before "
            "sampling. Can be passed multiple times."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading tests from {args.input_path} ...")
    tests = load_all_tests(args.input_path)
    print(f"  Loaded {len(tests)} tests total.")

    excluded_contexts: set[str] = set()
    for exclude_dataset in args.exclude_dataset:
        paths = load_context_paths(exclude_dataset)
        excluded_contexts.update(paths)
        print(f"  Loaded {len(paths)} excluded contexts from {exclude_dataset}.")

    input_contexts = set(group_by_context(tests))
    excluded_matching_contexts = input_contexts & excluded_contexts
    if excluded_contexts:
        print(
            f"  Excluding {len(excluded_matching_contexts)} matching input contexts "
            f"({len(excluded_contexts)} total context paths loaded)."
        )

    sampled, counts = sample_tests(
        tests=tests,
        target_questions=args.target_questions,
        target_contexts=args.target_contexts,
        min_per_context=args.min_per_context,
        seed=args.seed,
        exclude_contexts=excluded_contexts,
    )

    output = build_output(
        sampled=sampled,
        counts=counts,
        input_path=args.input_path,
        seed=args.seed,
        target_questions=args.target_questions,
        target_contexts=args.target_contexts,
        min_per_context=args.min_per_context,
        exclude_datasets=args.exclude_dataset,
        excluded_context_count=len(excluded_contexts),
        excluded_matching_context_count=len(excluded_matching_contexts),
    )

    out_path = args.output or default_output_path(
        target_questions=args.target_questions,
        target_contexts=args.target_contexts,
        min_per_context=args.min_per_context,
        seed=args.seed,
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(
        f"Sampled {len(sampled)} questions across {len(counts)} contexts "
        f"(seed={args.seed})."
    )
    print("Context breakdown:")
    for entry in output["contexts"]:
        print(f"  {entry['question_count']:3d}  {entry['file_path']}")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
