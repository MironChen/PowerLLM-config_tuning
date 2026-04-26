"""
Build paired mixed LegalBench-RAG datasets from multiple source JSON files.

For each input source, this script samples two disjoint context-level splits
using the same per-source quotas:

- `config`
- `validation`

Contexts selected for the config split are removed from the pool before the
validation split is sampled, so the two outputs do not share source documents.
Each --input file must contain a top-level "tests" list in LegalBench-RAG
format.
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_MIN_PER_CONTEXT = 10
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Loading & context grouping (aligned with dataset_cut3.py)
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

    return min(counts, key=lambda p: (-counts[p], p))


def group_by_context(tests: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for test in tests:
        primary = _primary_file_path(test)
        groups[primary].append(test)
    return dict(groups)


# ---------------------------------------------------------------------------
# Limit resolution
# ---------------------------------------------------------------------------


def _resolve_limits(
    n_sources: int,
    values: list[int] | None,
    flag_name: str,
) -> list[int | None]:
    """Broadcast a single limit to all sources, or validate one value per source."""
    if values is None:
        return [None] * n_sources
    if len(values) == 1:
        return [values[0]] * n_sources
    if len(values) != n_sources:
        raise ValueError(
            f"{flag_name}: expected 1 or {n_sources} integer(s), got {len(values)}: {values!r}"
        )
    return list(values)


def _task_name(path: Path) -> str:
    return path.stem


def _validate_unique_task_names(paths: list[Path]) -> None:
    """Reject duplicate task names so task_breakdown cannot be overwritten."""
    by_task: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        by_task[_task_name(path)].append(path)

    duplicates = {
        task: task_paths for task, task_paths in by_task.items() if len(task_paths) > 1
    }
    if not duplicates:
        return

    details = ", ".join(
        f"{task}: {[str(path) for path in task_paths]}"
        for task, task_paths in sorted(duplicates.items())
    )
    raise ValueError(
        "Duplicate input task names detected. Input basenames must be unique so "
        f"task_breakdown keys cannot be overwritten: {details}"
    )


def _stable_subseed(label: str) -> int:
    """Process-stable offset for per-source RNG (unlike built-in hash())."""
    return zlib.adler32(label.encode("utf-8")) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _select_context_keys(
    eligible: dict[str, list[dict[str, Any]]],
    target_contexts: int,
    rng: random.Random,
) -> list[str]:
    keys = list(eligible.keys())
    rng.shuffle(keys)
    keys.sort(key=lambda fp: len(eligible[fp]), reverse=True)
    return keys[:target_contexts]


def _sample_fixed_contexts(
    eligible: dict[str, list[dict[str, Any]]],
    selected_fps: list[str],
    target_questions: int,
    min_per_context: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Balanced sample across selected_fps (same min + leftover fill as dataset_cut3)."""
    target_contexts = len(selected_fps)
    floor = target_contexts * min_per_context
    if target_questions < floor:
        raise ValueError(
            f"max_questions ({target_questions}) must be >= "
            f"max_contexts * min_per_context ({floor})."
        )

    total_available = sum(len(eligible[fp]) for fp in selected_fps)
    if total_available < target_questions:
        raise ValueError(
            f"Selected {target_contexts} contexts only have {total_available} questions "
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


def sample_source(
    *,
    tests: list[dict[str, Any]],
    max_contexts: int,
    max_questions: int,
    min_per_context: int,
    rng: random.Random,
    source_label: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, int],
    list[dict[str, Any]],
    dict[str, int],
]:
    """
    Sample disjoint config/validation splits from one source file.

    Both splits use the same max_contexts/max_questions quota. Config contexts are
    selected first, then validation contexts are selected from the remaining
    pool so the two splits do not share source documents.
    """
    if min_per_context <= 0:
        raise ValueError("min_per_context must be a positive integer.")
    if max_contexts <= 0:
        raise ValueError(f"{source_label}: max_contexts must be positive.")
    if max_questions <= 0:
        raise ValueError(f"{source_label}: max_questions must be positive.")

    grouped = group_by_context(tests)
    eligible = {
        fp: cases
        for fp, cases in grouped.items()
        if len(cases) >= min_per_context
    }

    if not eligible:
        raise ValueError(
            f"{source_label}: no contexts with at least {min_per_context} questions."
        )

    if len(eligible) < max_contexts * 2:
        raise ValueError(
            f"{source_label}: need {max_contexts * 2} contexts with "
            f">= {min_per_context} questions each to build disjoint config/validation "
            f"splits, found {len(eligible)}."
        )

    config_rng = random.Random(rng.randrange(2**32))
    validation_rng = random.Random(rng.randrange(2**32))

    config_selected_fps = _select_context_keys(eligible, max_contexts, config_rng)
    remaining_eligible = {
        fp: cases for fp, cases in eligible.items() if fp not in set(config_selected_fps)
    }
    validation_selected_fps = _select_context_keys(
        remaining_eligible,
        max_contexts,
        validation_rng,
    )

    config_sampled, config_counts = _sample_fixed_contexts(
        eligible,
        config_selected_fps,
        max_questions,
        min_per_context,
        config_rng,
    )
    validation_sampled, validation_counts = _sample_fixed_contexts(
        remaining_eligible,
        validation_selected_fps,
        max_questions,
        min_per_context,
        validation_rng,
    )

    overlapping_contexts = set(config_counts) & set(validation_counts)
    if overlapping_contexts:
        raise ValueError(
            f"{source_label}: config and validation splits overlap on contexts: "
            f"{sorted(overlapping_contexts)!r}"
        )

    return (
        config_sampled,
        config_counts,
        validation_sampled,
        validation_counts,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def build_output(
    *,
    merged_tests: list[dict[str, Any]],
    task_breakdown: dict[str, Any],
    context_summary: list[dict[str, Any]],
    source_paths: list[Path],
    seed: int,
    min_per_context: int,
    per_source_sampling: list[dict[str, Any]],
    split_role: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "format": "legalbenchrag_mixed_split_v1",
        "source_inputs": [str(p.resolve()) for p in source_paths],
        "sampling": {
            "seed": seed,
            "min_per_context": min_per_context,
            "questions_sampled": len(merged_tests),
            "contexts_sampled": len(context_summary),
            "disjoint_by": "context",
            "per_source": per_source_sampling,
        },
        "contexts": context_summary,
        "tests": merged_tests,
        "task_breakdown": task_breakdown,
        "split_role": split_role,
    }
    return out


def default_output_path(
    *,
    total_questions: int,
    n_sources: int,
    seed: int,
    split_role: str,
) -> Path:
    name = f"legalbench_mix_{split_role}_q{total_questions}_{n_sources}src_seed{seed}.json"
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
        "--input",
        type=Path,
        nargs="+",
        required=True,
        metavar="PATH",
        help=(
            "One or more LegalBench-RAG JSON files. Each file is sampled twice "
            "using the same per-source quota: once for config and once for validation."
        ),
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Max contexts per input file, for each split. Pass one value to "
            "broadcast, or one value per --input."
        ),
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "Max questions per input file, for each split. Pass one value to "
            "broadcast, or one value per --input."
        ),
    )
    parser.add_argument(
        "--min-per-context",
        type=int,
        default=DEFAULT_MIN_PER_CONTEXT,
        help=(
            f"Minimum questions required for a context to be eligible for either "
            f"split (default: {DEFAULT_MIN_PER_CONTEXT})."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output JSON path for the config split.",
    )
    parser.add_argument(
        "--output-validation",
        type=Path,
        default=None,
        metavar="FILE",
        help="Output JSON path for the validation split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.input
    n = len(inputs)

    for p in inputs:
        if not p.is_file():
            raise FileNotFoundError(f"Input path is not a file: {p}")
    _validate_unique_task_names(inputs)

    max_contexts_list = _resolve_limits(n, args.max_contexts, "--max-contexts")
    max_questions_list = _resolve_limits(n, args.max_questions, "--max-questions")
    if any(value is None for value in max_contexts_list):
        raise ValueError("--max-contexts is required for dual-output split generation.")
    if any(value is None for value in max_questions_list):
        raise ValueError("--max-questions is required for dual-output split generation.")

    merged_config: list[dict[str, Any]] = []
    merged_validation: list[dict[str, Any]] = []
    task_breakdown_config: dict[str, Any] = {}
    task_breakdown_validation: dict[str, Any] = {}
    per_source_sampling_config: list[dict[str, Any]] = []
    per_source_sampling_validation: list[dict[str, Any]] = []
    global_counts_config: dict[str, int] = defaultdict(int)
    global_counts_validation: dict[str, int] = defaultdict(int)

    for i, path in enumerate(inputs):
        task = _task_name(path)
        rng = random.Random(args.seed + i * 1_000_003 + _stable_subseed(task))

        print(f"Loading {path} ...")
        tests = _load_file(path)
        print(f"  {len(tests)} tests in file.")

        mc, mq = max_contexts_list[i], max_questions_list[i]
        config_sampled, config_counts, validation_sampled, validation_counts = sample_source(
            tests=tests,
            max_contexts=mc,
            max_questions=mq,
            min_per_context=args.min_per_context,
            rng=rng,
            source_label=str(path),
        )

        config_contexts_detail = [
            {"file_path": fp, "question_count": c}
            for fp, c in sorted(config_counts.items(), key=lambda x: x[0])
        ]
        validation_contexts_detail = [
            {"file_path": fp, "question_count": c}
            for fp, c in sorted(validation_counts.items(), key=lambda x: x[0])
        ]
        task_breakdown_config[task] = {
            "questions": len(config_sampled),
            "contexts": len(config_counts),
            "contexts_detail": config_contexts_detail,
        }
        task_breakdown_validation[task] = {
            "questions": len(validation_sampled),
            "contexts": len(validation_counts),
            "contexts_detail": validation_contexts_detail,
        }
        per_source_sampling_config.append(
            {
                "path": str(path.resolve()),
                "task": task,
                "max_contexts": mc,
                "max_questions": mq,
                "questions_sampled": len(config_sampled),
                "contexts_sampled": len(config_counts),
            }
        )
        per_source_sampling_validation.append(
            {
                "path": str(path.resolve()),
                "task": task,
                "max_contexts": mc,
                "max_questions": mq,
                "questions_sampled": len(validation_sampled),
                "contexts_sampled": len(validation_counts),
            }
        )

        for fp, c in config_counts.items():
            global_counts_config[fp] += c
        for fp, c in validation_counts.items():
            global_counts_validation[fp] += c

        merged_config.extend(config_sampled)
        merged_validation.extend(validation_sampled)

    rng_merge_config = random.Random(args.seed)
    rng_merge_validation = random.Random(args.seed + 1)
    rng_merge_config.shuffle(merged_config)
    rng_merge_validation.shuffle(merged_validation)

    context_summary_config = [
        {"file_path": fp, "question_count": c}
        for fp, c in sorted(global_counts_config.items(), key=lambda x: x[0])
    ]
    context_summary_validation = [
        {"file_path": fp, "question_count": c}
        for fp, c in sorted(global_counts_validation.items(), key=lambda x: x[0])
    ]

    output_config = build_output(
        merged_tests=merged_config,
        task_breakdown=task_breakdown_config,
        context_summary=context_summary_config,
        source_paths=inputs,
        seed=args.seed,
        min_per_context=args.min_per_context,
        per_source_sampling=per_source_sampling_config,
        split_role="config",
    )
    output_validation = build_output(
        merged_tests=merged_validation,
        task_breakdown=task_breakdown_validation,
        context_summary=context_summary_validation,
        source_paths=inputs,
        seed=args.seed,
        min_per_context=args.min_per_context,
        per_source_sampling=per_source_sampling_validation,
        split_role="validation",
    )

    out_path_config = args.output_config or default_output_path(
        total_questions=len(merged_config),
        n_sources=n,
        seed=args.seed,
        split_role="config",
    )
    out_path_validation = args.output_validation or default_output_path(
        total_questions=len(merged_validation),
        n_sources=n,
        seed=args.seed,
        split_role="validation",
    )
    with out_path_config.open("w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)
    with out_path_validation.open("w", encoding="utf-8") as f:
        json.dump(output_validation, f, indent=2, ensure_ascii=False)

    print(
        f"\nConfig: {len(merged_config)} questions across "
        f"{len(context_summary_config)} contexts."
    )
    print(
        f"Validation: {len(merged_validation)} questions across "
        f"{len(context_summary_validation)} contexts."
    )
    print("Per-task breakdown:")
    for task in sorted(task_breakdown_config):
        config_info = task_breakdown_config[task]
        validation_info = task_breakdown_validation[task]
        print(
            f"  {task}: "
            f"config {config_info['questions']} q / {config_info['contexts']} contexts, "
            f"validation {validation_info['questions']} q / {validation_info['contexts']} contexts"
        )
    print(f"\nSaved config to {out_path_config}")
    print(f"Saved validation to {out_path_validation}")


if __name__ == "__main__":
    main()
