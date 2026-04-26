"""Benchmark evaluation metrics for retrieval quality assessment."""
from __future__ import annotations

from typing import Any

from benchmark.benchmark_utils import (
    _chunk_covers_span,
    filter_valid_expected_answer_spans,
)

DEFAULT_RECALL_KS: tuple[int, ...] = (10, 20)
DEFAULT_HIT_KS: tuple[int, ...] = (1, 3, 5)
DEFAULT_PRECISION_KS: tuple[int, ...] = (5, 10, 20)

# Coverage classification constants
COVERAGE_YES = "Yes"
COVERAGE_NO = "No"
COVERAGE_PARTIAL = "partial"
COVERAGE_NA = "N/A"

# Miss diagnostic classification constants
MISS_FULLY_COVERED = "fully_covered"
MISS_BOUNDARY_SPLIT = "boundary_split"
MISS_TRUE_MISS = "true_miss"
MISS_MIXED_MISS = "mixed_miss"
MISS_NO_VALID_SPANS = "no_valid_spans"


def _select_top_k_chunks(
    retrieved_chunks: list[dict],
    *,
    k: int,
) -> list[dict]:
    """Return the top-k chunks using explicit rank when available."""
    if k <= 0:
        return []

    return [
        chunk
        for index, chunk in enumerate(retrieved_chunks, start=1)
        if (
            isinstance(chunk.get("rank"), int) and chunk["rank"] <= k
        ) or (
            not isinstance(chunk.get("rank"), int) and index <= k
        )
    ]


def _chunk_bounds(chunk: dict) -> tuple[int, int] | None:
    """Return (start, end) for a chunk that has valid integer bounds, or None."""
    start = chunk.get("start")
    end = chunk.get("end")
    if isinstance(start, int) and isinstance(end, int) and start >= 0 and end > start:
        return (start, end)
    return None


def _span_is_covered_by_chunks(
    answer_start: int,
    answer_end: int,
    chunks: list[dict],
) -> bool:
    return any(
        _chunk_covers_span(
            chunk,
            answer_start=answer_start,
            answer_end=answer_end,
        )
        for chunk in chunks
    )


def _chunk_overlaps_span(
    chunk: dict,
    *,
    answer_start: int,
    answer_end: int,
) -> bool:
    bounds = _chunk_bounds(chunk)
    if bounds is None:
        return False
    start, end = bounds
    return start < answer_end and end > answer_start


def _calculate_span_overlap_ratio(
    chunk: dict,
    *,
    answer_start: int,
    answer_end: int,
) -> float:
    bounds = _chunk_bounds(chunk)
    if bounds is None:
        return 0.0
    start, end = bounds
    overlap = max(0, min(end, answer_end) - max(start, answer_start))
    span_length = answer_end - answer_start
    if span_length <= 0:
        return 0.0
    return overlap / span_length


def _calculate_precision_at_k(
    retrieved_chunks: list[dict],
    valid_spans: list[dict],
    *,
    k: int,
) -> float | None:
    """
    Calculate traditional retrieval precision@k.

    Precision = relevant retrieved chunks / retrieved chunks. A chunk is relevant
    when it fully covers at least one valid answer span.
    """
    top_k_chunks = _select_top_k_chunks(retrieved_chunks, k=k)

    if not top_k_chunks or not valid_spans:
        return None

    relevant_chunks = 0
    for chunk in top_k_chunks:
        if any(
            _chunk_covers_span(
                chunk,
                answer_start=span["answer_start"],
                answer_end=span["answer_end"],
            )
            for span in valid_spans
        ):
            relevant_chunks += 1

    return relevant_chunks / len(top_k_chunks)


def _calculate_f1_score(
    recall: float | None,
    precision: float | None,
) -> float | None:
    """Return the harmonic mean of recall and precision when both exist."""
    if recall is None or precision is None:
        return None
    if recall + precision == 0:
        return 0.0
    return (2 * recall * precision) / (recall + precision)


def aggregate_mean(per_question_values: list[float | None]) -> float | None:
    """Aggregate per-question metric values into a mean, skipping None."""
    valid_values = [v for v in per_question_values if v is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def aggregate_retrieval_results(
    per_question_results: list[dict[str, Any]],
    *,
    recall_ks: tuple[int, ...],
    hit_ks: tuple[int, ...],
    precision_ks: tuple[int, ...],
) -> dict[str, Any]:
    """
    Aggregate per-question retrieval outputs into the shared summary shape.

    Uses single-pass aggregation for efficiency with large result sets.
    """
    total_questions = len(per_question_results)

    # Initialize accumulators for single-pass aggregation
    answerable_questions = 0
    fully_covered_count = 0
    partial_covered_count = 0
    not_covered_count = 0

    # Metric accumulators: {k: [values]} for each metric type
    recall_values: dict[str, list[float | None]] = {
        f"at_{k}": [] for k in recall_ks
    }
    recall_values["at_final_k"] = []

    hit_values: dict[str, list[float | None]] = {f"at_{k}": [] for k in hit_ks}

    precision_values: dict[str, list[float | None]] = {
        f"at_{k}": [] for k in precision_ks
    }
    precision_values["at_final_k"] = []
    shared_f1_keys = sorted(
        set(recall_values).intersection(precision_values),
        key=lambda key: (key != "at_final_k", key),
    )
    f1_values: dict[str, list[float | None]] = {
        key: [] for key in shared_f1_keys
    }

    mrr_values: list[float | None] = []
    fully_covered_values: list[float | None] = []
    overlap_ratio_values: list[float | None] = []
    failure_analyses: list[dict[str, Any] | None] = []

    # Single pass over all results
    for item in per_question_results:
        has_valid = item.get("has_valid_gold_spans", False)

        if has_valid:
            answerable_questions += 1
            coverage = item.get("coverage")
            if coverage == COVERAGE_YES:
                fully_covered_count += 1
            elif coverage == COVERAGE_PARTIAL:
                partial_covered_count += 1
            elif coverage == COVERAGE_NO:
                not_covered_count += 1

        retrieval_metrics = item.get("retrieval_metrics") or {}

        # Collect recall values
        recall = retrieval_metrics.get("recall") or {}
        for key in recall_values:
            recall_values[key].append(recall.get(key))

        # Collect hit values
        hit = retrieval_metrics.get("hit") or {}
        for key in hit_values:
            hit_values[key].append(hit.get(key))

        # Collect precision values
        precision = retrieval_metrics.get("precision") or {}
        for key in precision_values:
            precision_values[key].append(precision.get(key))
        f1 = retrieval_metrics.get("f1") or {}
        for key in f1_values:
            f1_values[key].append(f1.get(key))

        # Collect ranking and context quality values
        ranking = retrieval_metrics.get("ranking") or {}
        mrr_values.append(ranking.get("mrr"))

        context_quality = retrieval_metrics.get("context_quality") or {}
        fully_covered_values.append(context_quality.get("fully_covered_at_final_k"))
        overlap_ratio_values.append(
            context_quality.get("mean_overlap_ratio_at_final_k")
        )

        failure_analyses.append(item.get("failure_analysis"))

    partial_or_better_count = fully_covered_count + partial_covered_count
    no_answer_questions = total_questions - answerable_questions

    # Build summaries using collected values
    recall_summary = {f"{k}_mean": aggregate_mean(v) for k, v in recall_values.items()}
    hit_summary = {f"{k}_mean": aggregate_mean(v) for k, v in hit_values.items()}
    precision_summary = {
        f"{k}_mean": aggregate_mean(v) for k, v in precision_values.items()
    }
    f1_summary = {f"{k}_mean": aggregate_mean(v) for k, v in f1_values.items()}

    miss_diagnostic_summary = aggregate_miss_diagnostics(failure_analyses)

    return {
        "total_questions": total_questions,
        "answerable_questions": answerable_questions,
        "no_answer_questions": no_answer_questions,
        "metrics": {
            "recall": recall_summary,
            "hit": hit_summary,
            "precision": precision_summary,
            "f1": f1_summary,
            "ranking": {
                "mrr_mean": aggregate_mean(mrr_values),
            },
            "context_quality": {
                "fully_covered_at_final_k_mean": aggregate_mean(fully_covered_values),
                "mean_overlap_ratio_at_final_k_mean": aggregate_mean(
                    overlap_ratio_values
                ),
            },
        },
        "coverage": {
            "fully_covered": {
                "count": fully_covered_count,
                "rate": (
                    fully_covered_count / answerable_questions
                    if answerable_questions
                    else 0.0
                ),
            },
            "partial_covered": {
                "count": partial_covered_count,
                "rate": (
                    partial_covered_count / answerable_questions
                    if answerable_questions
                    else 0.0
                ),
            },
            "not_covered": {
                "count": not_covered_count,
                "rate": (
                    not_covered_count / answerable_questions
                    if answerable_questions
                    else 0.0
                ),
            },
            "partial_or_better": {
                "count": partial_or_better_count,
                "rate": (
                    partial_or_better_count / answerable_questions
                    if answerable_questions
                    else 0.0
                ),
            },
        },
        "failure_analysis": {
            "diagnosed_miss_count": miss_diagnostic_summary["diagnosed_miss_count"],
            "question_level": {
                "boundary_split": {
                    "count": miss_diagnostic_summary["boundary_split_count"],
                    "rate": miss_diagnostic_summary["boundary_split_rate"],
                },
                "true_miss": {
                    "count": miss_diagnostic_summary["true_miss_count"],
                    "rate": miss_diagnostic_summary["true_miss_rate"],
                },
                "mixed_miss": {
                    "count": miss_diagnostic_summary["mixed_miss_count"],
                    "rate": miss_diagnostic_summary["mixed_miss_rate"],
                },
            },
            "span_level": {
                "boundary_split": {
                    "count": miss_diagnostic_summary["boundary_split_span_count"],
                    "rate": miss_diagnostic_summary["boundary_split_span_rate"],
                },
                "true_miss": {
                    "count": miss_diagnostic_summary["true_miss_span_count"],
                    "rate": miss_diagnostic_summary["true_miss_span_rate"],
                },
                "total_missing_spans": miss_diagnostic_summary[
                    "diagnosed_missing_span_count"
                ],
            },
        },
    }


def _format_k_key(k: int) -> str:
    return f"at_{k}"


def evaluate_retrieval_question(
    expected_answer_annotations: list[dict],
    retrieved_chunks: list[dict],
    *,
    final_k: int,
    final_retrieved_chunks: list[dict] | None = None,
    recall_ks: tuple[int, ...] = DEFAULT_RECALL_KS,
    hit_ks: tuple[int, ...] = DEFAULT_HIT_KS,
    precision_ks: tuple[int, ...] = DEFAULT_PRECISION_KS,
) -> dict[str, Any]:
    """Compute all retrieval metrics for a single question in one pass.

    Avoids redundant calls to filter_valid_expected_answer_spans and
    _select_top_k_chunks by pre-computing shared intermediates.
    """
    recall_k_values = tuple(dict.fromkeys(recall_ks))
    hit_k_values = tuple(dict.fromkeys(hit_ks))
    precision_k_values = tuple(dict.fromkeys(precision_ks))
    top_k_chunks_cache = {
        k: _select_top_k_chunks(retrieved_chunks, k=k)
        for k in set(recall_k_values + hit_k_values + precision_k_values + (final_k,))
    }
    top_final_k_chunks = (
        final_retrieved_chunks
        if final_retrieved_chunks is not None
        else top_k_chunks_cache[final_k]
    )

    valid_spans = filter_valid_expected_answer_spans(expected_answer_annotations)
    if not valid_spans:
        return {
            "has_valid_gold_spans": False,
            "coverage": COVERAGE_NA,
            "retrieval_metrics": {
                "recall": {
                    _format_k_key(k): None for k in recall_k_values
                } | {
                    "at_final_k": None,
                },
                "hit": {
                    _format_k_key(k): None for k in hit_k_values
                },
                "precision": {
                    _format_k_key(k): None for k in precision_k_values
                } | {
                    "at_final_k": None,
                },
                "f1": {
                    _format_k_key(k): None
                    for k in sorted(set(recall_k_values).intersection(precision_k_values))
                } | {
                    "at_final_k": None,
                },
                "ranking": {
                    "mrr": None,
                },
                "context_quality": {
                    "fully_covered_at_final_k": None,
                    "mean_overlap_ratio_at_final_k": None,
                },
            },
            "failure_analysis": {
                "classification": MISS_NO_VALID_SPANS,
                "missing_span_count": 0,
                "boundary_split_span_count": 0,
                "true_miss_span_count": 0,
            },
        }

    precision_at_k = {
        _format_k_key(k): _calculate_precision_at_k(
            retrieved_chunks,
            valid_spans,
            k=k,
        )
        for k in precision_k_values
    }
    precision_at_final_k = _calculate_precision_at_k(
        top_final_k_chunks,
        valid_spans,
        k=final_k,
    )
    recall_at_final_k = covered_final_k = 0
    total_spans = len(valid_spans)

    # Single pass: for each span, check coverage across all chunk lists.
    covered_at_k = {k: 0 for k in recall_k_values}
    hit_at_k = {k: 0 for k in hit_k_values}
    best_overlap_ratios: list[float] = []
    first_cover_rank: int | None = None

    boundary_split_span_count = 0
    true_miss_span_count = 0

    for annotation in valid_spans:
        a_start = annotation["answer_start"]
        a_end = annotation["answer_end"]

        in_final_k = _span_is_covered_by_chunks(a_start, a_end, top_final_k_chunks)

        if in_final_k:
            covered_final_k += 1

        for k in recall_k_values:
            if _span_is_covered_by_chunks(
                a_start,
                a_end,
                top_k_chunks_cache[k],
            ):
                covered_at_k[k] += 1

        for k in hit_k_values:
            if _span_is_covered_by_chunks(
                a_start,
                a_end,
                top_k_chunks_cache[k],
            ):
                hit_at_k[k] = 1

        # Best overlap ratio against final_k chunks
        best_ratio = 0.0
        for chunk in top_final_k_chunks:
            best_ratio = max(
                best_ratio,
                _calculate_span_overlap_ratio(chunk, answer_start=a_start, answer_end=a_end),
            )
        best_overlap_ratios.append(best_ratio)

        # MRR: find first chunk that covers this span (across all chunks)
        if first_cover_rank is None:
            for rank, chunk in enumerate(retrieved_chunks, start=1):
                if _chunk_covers_span(chunk, answer_start=a_start, answer_end=a_end):
                    first_cover_rank = rank
                    break

        # Miss diagnostic: for spans not covered by final_k
        if not in_final_k:
            overlapping_chunks = [
                chunk for chunk in top_final_k_chunks
                if _chunk_overlaps_span(chunk, answer_start=a_start, answer_end=a_end)
            ]
            if not overlapping_chunks:
                true_miss_span_count += 1
            else:
                merged_start = min(c["start"] for c in overlapping_chunks)
                merged_end = max(c["end"] for c in overlapping_chunks)
                if merged_start <= a_start and merged_end >= a_end:
                    boundary_split_span_count += 1
                else:
                    true_miss_span_count += 1

    # Derive coverage classification
    if covered_final_k == 0:
        coverage = COVERAGE_NO
    elif covered_final_k == total_spans:
        coverage = COVERAGE_YES
    else:
        coverage = COVERAGE_PARTIAL

    # Miss diagnostic classification
    missing_span_count = boundary_split_span_count + true_miss_span_count
    if missing_span_count == 0:
        miss_classification = MISS_FULLY_COVERED
    elif boundary_split_span_count > 0 and true_miss_span_count > 0:
        miss_classification = MISS_MIXED_MISS
    elif true_miss_span_count > 0:
        miss_classification = MISS_TRUE_MISS
    else:
        miss_classification = MISS_BOUNDARY_SPLIT

    recall_at_k = {
        _format_k_key(k): covered_at_k[k] / total_spans
        for k in recall_k_values
    }
    recall_at_final_k = covered_final_k / total_spans
    f1_at_k = {
        key: _calculate_f1_score(recall_at_k.get(key), precision_at_k.get(key))
        for key in precision_at_k
        if key in recall_at_k
    }
    f1_at_final_k = _calculate_f1_score(recall_at_final_k, precision_at_final_k)

    return {
        "has_valid_gold_spans": True,
        "coverage": coverage,
        "retrieval_metrics": {
            "recall": {
                **recall_at_k,
                "at_final_k": recall_at_final_k,
            },
            "hit": {
                **{
                    _format_k_key(k): float(hit_at_k[k])
                    for k in hit_k_values
                },
            },
            "precision": {
                **precision_at_k,
                "at_final_k": precision_at_final_k,
            },
            "f1": {
                **f1_at_k,
                "at_final_k": f1_at_final_k,
            },
            "ranking": {
                "mrr": (
                    (1.0 / first_cover_rank)
                    if first_cover_rank is not None
                    else 0.0
                ),
            },
            "context_quality": {
                "fully_covered_at_final_k": (
                    1.0 if covered_final_k == total_spans else 0.0
                ),
                "mean_overlap_ratio_at_final_k": (
                    sum(best_overlap_ratios) / len(best_overlap_ratios)
                ),
            },
        },
        "failure_analysis": {
            "classification": miss_classification,
            "missing_span_count": missing_span_count,
            "boundary_split_span_count": boundary_split_span_count,
            "true_miss_span_count": true_miss_span_count,
        },
    }


def aggregate_miss_diagnostics(
    per_question_diagnostics: list[dict[str, int | float | str] | None],
) -> dict[str, int | float]:
    """Aggregate question-level miss diagnostics for not-fully-covered questions."""
    valid_diagnostics = [
        diagnostic
        for diagnostic in per_question_diagnostics
        if diagnostic is not None
    ]
    diagnostic_questions = [
        diagnostic
        for diagnostic in valid_diagnostics
        if diagnostic.get("classification")
        in {
            MISS_BOUNDARY_SPLIT,
            MISS_TRUE_MISS,
            MISS_MIXED_MISS,
        }
    ]
    total_misses = len(diagnostic_questions)

    boundary_split_count = sum(
        1
        for diagnostic in diagnostic_questions
        if diagnostic.get("classification") == MISS_BOUNDARY_SPLIT
    )
    true_miss_count = sum(
        1
        for diagnostic in diagnostic_questions
        if diagnostic.get("classification") == MISS_TRUE_MISS
    )
    mixed_miss_count = sum(
        1
        for diagnostic in diagnostic_questions
        if diagnostic.get("classification") == MISS_MIXED_MISS
    )
    boundary_split_span_count = sum(
        int(diagnostic.get("boundary_split_span_count", 0) or 0)
        for diagnostic in diagnostic_questions
    )
    true_miss_span_count = sum(
        int(diagnostic.get("true_miss_span_count", 0) or 0)
        for diagnostic in diagnostic_questions
    )
    total_missing_spans = boundary_split_span_count + true_miss_span_count

    return {
        "boundary_split_count": boundary_split_count,
        "boundary_split_rate": (
            boundary_split_count / total_misses if total_misses else 0.0
        ),
        "true_miss_count": true_miss_count,
        "true_miss_rate": true_miss_count / total_misses if total_misses else 0.0,
        "mixed_miss_count": mixed_miss_count,
        "mixed_miss_rate": mixed_miss_count / total_misses if total_misses else 0.0,
        "diagnosed_miss_count": total_misses,
        "boundary_split_span_count": boundary_split_span_count,
        "boundary_split_span_rate": (
            boundary_split_span_count / total_missing_spans
            if total_missing_spans
            else 0.0
        ),
        "true_miss_span_count": true_miss_span_count,
        "true_miss_span_rate": (
            true_miss_span_count / total_missing_spans
            if total_missing_spans
            else 0.0
        ),
        "diagnosed_missing_span_count": total_missing_spans,
    }
