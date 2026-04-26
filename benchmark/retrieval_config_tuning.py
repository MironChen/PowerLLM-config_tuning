"""
Retrieval Configuration Tuning Script using Optuna.

This script performs hyperparameter optimization for the RAG retrieval pipeline.
By default it tunes chunk size plus retrieval candidate-pool sizes to maximize
retrieval quality under a fixed final retrieval window.

Usage:
    # Quick test run (10 trials, default limited dataset)
    python -m benchmark.retrieval_config_tuning --n-trials 10

    # Full tuning (200 trials, complete dataset)
    python -m benchmark.retrieval_config_tuning --n-trials 200

    # Resume from previous run
    python -m benchmark.retrieval_config_tuning --n-trials 100 --storage sqlite:///optuna.db

The script will:
1. Define search spaces for tunable parameters
2. Run benchmark trials with sampled configurations
3. Track results in SQLite database (supports interruption/resumption)
4. Output best parameters and summary artifacts

Adjustable Areas (marked with [ADJUST] comments):
- Search space bounds for chunking and retrieval candidate pools
- Objective function weights
- Pruning thresholds
- Benchmark scope for speed/accuracy tradeoff
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import optuna
from optuna.samplers import GridSampler, RandomSampler, TPESampler
from optuna.pruners import MedianPruner, NopPruner

# # Optuna imports for hyperparameter optimization
# try:
#     import optuna
#     from optuna.samplers import TPESampler
#     from optuna.pruners import MedianPruner
# except ImportError:
#     print("Error: optuna is required. Install with: pip install optuna")
#     sys.exit(1)

from dotenv import load_dotenv

load_dotenv()

from benchmark.benchmark_pipeline import (
    get_answer,
    get_or_create_persistent_vector_store,
    release_persistent_vector_store,
)
from benchmark.benchmark_metrics import (
    DEFAULT_HIT_KS,
    DEFAULT_PRECISION_KS,
    DEFAULT_RECALL_KS,
    aggregate_retrieval_results,
    evaluate_retrieval_question,
)
from benchmark.benchmark_utils import (
    atomic_write_json,
    build_bm25_cache_key,
    build_cuad_answer_annotations,
    calculate_latency,
    precompute_bm25_queries_if_needed,
)
from benchmark.chunking_strategies import (
    DEFAULT_CHUNKING_STRATEGY,
    ChunkingStrategy,
)
import powerllm.retrieval.chunker as chunker
import powerllm.retrieval.pipeline as pipeline
from powerllm.models.model_resolver import check_chat_model_connection
# =============================================================================
# [ADJUST] Default Configuration - Modify these defaults for your use case
# =============================================================================
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
if not CLOUDFLARE_ACCOUNT_ID:
    raise ValueError("CLOUDFLARE_ACCOUNT_ID is not set")
CLOUDFLARE_AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")
if not CLOUDFLARE_AUTH_TOKEN:
    raise ValueError("CLOUDFLARE_AUTH_TOKEN is not set")

# Paths
DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent
    / "legalbench_mixed_config_q200_c20_seed42.json"
)
DEFAULT_OUTPUT_DIR = Path("benchmark_results/tuning")
DEFAULT_VECTOR_CACHE_DIR = Path("benchmark/chroma_cache")
DEFAULT_BM25_QUERY_CACHE_DIR = Path("benchmark/query_cache")
DEFAULT_STORAGE_URL = "sqlite:///benchmark/config_tuning.db"

# Fixed model configurations (not tuned)
EMBEDDING_MODEL_ID = "Qwen3-Embedding-0.6B-4bit-DWQ"
CHAT_MODEL_ID = "gemini-3.1-flash-lite"
BM25_QUERY_LLM_MODE = "concurrent"

FIXED_CHUNK_SIZE = 2000
FIXED_CHUNK_OVERLAP = 400
FINAL_CONTEXT_CHAR_BUDGET = 22_000

RETRIEVAL_K_SIMILARITY_MIN = 4
RETRIEVAL_K_SIMILARITY_MAX = 16
RETRIEVAL_K_BM25_MIN = 20
RETRIEVAL_K_BM25_MAX = 100
RETRIEVAL_K_FINAL_MIN = 4
RETRIEVAL_K_FINAL_MAX = 16

CHUNK_SIZE_CHOICES: tuple[int, ...] = (
    256,
    512,
    1000,
    1500,
    2000,
    2100,
)
K_SET_CHOICES: dict[str, dict[str, int]] = {
    "A": {"similarity_k": 8, "bm25_k": 61},
    "B": {"similarity_k": 7, "bm25_k": 49},
    "C": {"similarity_k": 13, "bm25_k": 81},
}
SEARCH_SPACE_PRESETS: dict[str, dict[str, Any]] = {
    "fixed_chunking_2000": {
        "fixed_chunk_size": 2000,
        "fixed_chunk_overlap": 400,
        "retrieval_similarity_min": RETRIEVAL_K_SIMILARITY_MIN,
        "retrieval_similarity_max": RETRIEVAL_K_SIMILARITY_MAX,
        "retrieval_bm25_min": RETRIEVAL_K_BM25_MIN,
        "retrieval_bm25_max": RETRIEVAL_K_BM25_MAX,
        "final_k": {"type": "int", "low": RETRIEVAL_K_FINAL_MIN, "high": RETRIEVAL_K_FINAL_MAX},
        "fixed_rrf_k": 60,
        "bm25_query_cache_k": 50,
        "density_precision_target": 0.05,
        "recall_ks": DEFAULT_RECALL_KS,
        "hit_ks": DEFAULT_HIT_KS,
        "precision_ks": DEFAULT_PRECISION_KS,
    },
    "empirical_grid_baseline": {
        "chunk_size_choices": (512, 1000, 2000),
        "chunk_overlap_ratio": {"type": "fixed", "value": 0.20},
        "similarity_k": {"type": "int", "low": 4, "high": 12, "step": 4},
        "bm25_k": {"type": "int", "low": 40, "high": 80, "step": 20},
        "final_k": {"type": "int", "low": 8, "high": 14, "step": 2},
        "fixed_rrf_k": 60,
        "bm25_query_cache_k": 50,
        "density_precision_target": 0.05,
        "recall_ks": DEFAULT_RECALL_KS,
        "hit_ks": DEFAULT_HIT_KS,
        "precision_ks": DEFAULT_PRECISION_KS,
    },
    "fixed_chunking_1000": {
        "fixed_chunk_size": 1000,
        "fixed_chunk_overlap": 200,
        "retrieval_similarity_min": RETRIEVAL_K_SIMILARITY_MIN,
        "retrieval_similarity_max": RETRIEVAL_K_SIMILARITY_MAX,
        "retrieval_bm25_min": RETRIEVAL_K_BM25_MIN,
        "retrieval_bm25_max": RETRIEVAL_K_BM25_MAX,
        "final_k": {"type": "int", "low": RETRIEVAL_K_FINAL_MIN, "high": RETRIEVAL_K_FINAL_MAX},
        "fixed_rrf_k": 60,
        "bm25_query_cache_k": 50,
        "density_precision_target": 0.05,
        "recall_ks": DEFAULT_RECALL_KS,
        "hit_ks": DEFAULT_HIT_KS,
        "precision_ks": DEFAULT_PRECISION_KS,
    },
    "fixed_chunking_1500": {
        "fixed_chunk_size": 1500,
        "fixed_chunk_overlap": 300,
        "retrieval_similarity_min": RETRIEVAL_K_SIMILARITY_MIN,
        "retrieval_similarity_max": RETRIEVAL_K_SIMILARITY_MAX,
        "retrieval_bm25_min": RETRIEVAL_K_BM25_MIN,
        "retrieval_bm25_max": RETRIEVAL_K_BM25_MAX,
        "final_k": {"type": "int", "low": RETRIEVAL_K_FINAL_MIN, "high": RETRIEVAL_K_FINAL_MAX},
        "fixed_rrf_k": 60,
        "bm25_query_cache_k": 50,
        "density_precision_target": 0.05,
        "recall_ks": DEFAULT_RECALL_KS,
        "hit_ks": DEFAULT_HIT_KS,
        "precision_ks": DEFAULT_PRECISION_KS,
    },
    "baseline_kset": {
        "chunk_size_choices": CHUNK_SIZE_CHOICES,
        "chunk_overlap_ratio": {"type": "fixed", "value": 0.20},
        "k_set_choices": K_SET_CHOICES,
        "fixed_similarity_k": 9,
        "fixed_bm25_k": 39,
        "fixed_final_k": 10,
        "fixed_rrf_k": 60,
        "bm25_query_cache_k": 50,
        "density_precision_target": 0.05,
        "recall_ks": DEFAULT_RECALL_KS,
        "hit_ks": DEFAULT_HIT_KS,
        "precision_ks": DEFAULT_PRECISION_KS,
    },
    "tune_chunking_and_final_k": {
        "chunk_size_choices": CHUNK_SIZE_CHOICES,
        "chunk_overlap_ratio": {"type": "fixed", "value": 0.20},
        "fixed_similarity_k": 9,
        "fixed_bm25_k": 39,
        "final_k": {"type": "int", "low": RETRIEVAL_K_FINAL_MIN, "high": RETRIEVAL_K_FINAL_MAX},
        "fixed_rrf_k": 60,
        "bm25_query_cache_k": 50,
        "density_precision_target": 0.05,
        "recall_ks": DEFAULT_RECALL_KS,
        "hit_ks": DEFAULT_HIT_KS,
        "precision_ks": DEFAULT_PRECISION_KS,
    },
    "tpe_wide": {
        "chunk_size": {"type": "int", "low": 2000, "high": 9000, "step": 250},
        "chunk_overlap_ratio": {
            "type": "float",
            "low": 0.10,
            "high": 0.35,
            "step": 0.05,
        },
        "similarity_k": {"type": "int", "low": 4, "high": 24},
        "bm25_k": {"type": "int", "low": 20, "high": 140, "step": 5},
        "rrf_k": {"type": "int", "low": 20, "high": 120, "step": 10},
        "fixed_similarity_k": 9,
        "fixed_bm25_k": 39,
        "fixed_final_k": 10,
        "bm25_query_cache_k": 50,
        "density_precision_target": 0.05,
        "recall_ks": DEFAULT_RECALL_KS,
        "hit_ks": DEFAULT_HIT_KS,
        "precision_ks": DEFAULT_PRECISION_KS,
    },
    "small_chunks": {
        "chunk_size": {"type": "int", "low": 500, "high": 3000, "step": 250},
        "chunk_overlap_ratio": {
            "type": "float",
            "low": 0.10,
            "high": 0.35,
            "step": 0.05,
        },
        "similarity_k": {"type": "int", "low": 4, "high": 24},
        "bm25_k": {"type": "int", "low": 20, "high": 140, "step": 5},
        "rrf_k": {"type": "int", "low": 20, "high": 120, "step": 10},
        "fixed_similarity_k": 9,
        "fixed_bm25_k": 39,
        "fixed_final_k": 10,
        "bm25_query_cache_k": 50,
        "density_precision_target": 0.05,
        "recall_ks": DEFAULT_RECALL_KS,
        "hit_ks": DEFAULT_HIT_KS,
        "precision_ks": DEFAULT_PRECISION_KS,
    },
}
SEARCH_SPACE_PRESET_CHOICES: tuple[str, ...] = tuple(SEARCH_SPACE_PRESETS)
SEARCH_METHOD_CHOICES: tuple[str, ...] = ("tpe", "grid", "random")

for preset_name, preset_config in SEARCH_SPACE_PRESETS.items():
    if "rrf_k" in preset_config and "fixed_rrf_k" in preset_config:
        raise ValueError(
            f"Invalid search space '{preset_name}': specify either rrf_k or "
            "fixed_rrf_k, not both."
        )
    if "retrieval_similarity_min" not in preset_config:
        continue
    if (
        preset_config["final_k"]["high"]
        > preset_config["retrieval_similarity_min"] + preset_config["retrieval_bm25_min"]
    ):
        raise ValueError(
            f"Invalid search space '{preset_name}': final_k.high must be <= "
            "retrieval_similarity_min + retrieval_bm25_min."
        )

# =============================================================================
# [ADJUST] Benchmark Scope - Reduce for faster iteration during tuning
# =============================================================================

# Number of contracts to evaluate per trial
# Lower = faster trials, less accurate per-trial results
# Higher = slower trials, more reliable ranking of configurations
DEFAULT_LIMIT_CONTRACTS = 4

# Number of questions per contract
DEFAULT_LIMIT_QUESTIONS_PER_CONTRACT = 5

# Retrieval mode to tune (can be made categorical if you want to compare modes)
RETRIEVAL_MODE = "hybrid"

# System prompt for answer generation during tuning
TUNING_SYSTEM_PROMPT = (
    "Answer using only the exact text span(s) from the context. "
    "Do not explain. Do not cite sections. "
    "If no answer is found, return: NOT FOUND"
)


def load_legalbench_tuning_dataset(
    dataset_path: Path,
    *,
    limit_contracts: int = -1,
    limit_questions_per_contract: int = -1,
) -> list[dict[str, Any]]:
    """Load LegalBench-RAG benchmark JSON into the grouped tuning dataset shape."""
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_dataset = json.load(f)

    corpus_dir = dataset_path.resolve().parent / "corpus"
    grouped_records: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for index, record in enumerate(raw_dataset["tests"]):
        snippets = record["snippets"]
        title = snippets[0]["file_path"]
        context_path = corpus_dir / title
        context = context_path.read_text(encoding="utf-8")

        answers: list[dict[str, Any]] = []
        for snippet in snippets:
            span = snippet["span"]
            answers.append(
                {
                    "answer_start": span[0],
                    "text": snippet["answer"],
                }
            )

        key = (title, context)
        grouped_records.setdefault(key, []).append(
            {
                "id": str(record.get("id") or f"{title}_{index}"),
                "question": record["query"],
                "answers": answers,
                "is_impossible": False,
            }
        )

    dataset = [
        {
            "title": title,
            "context": context,
            "qas": (
                qas[:limit_questions_per_contract]
                if limit_questions_per_contract >= 0
                else qas
            ),
        }
        for (title, context), qas in grouped_records.items()
    ]

    if limit_contracts >= 0:
        dataset = dataset[:limit_contracts]

    return dataset

# =============================================================================
# [ADJUST] Search Space Definition - Modify bounds based on your constraints
# =============================================================================

def _suggest_int_from_spec(
    trial: optuna.Trial,
    name: str,
    spec: dict[str, Any],
) -> int:
    """Sample an integer parameter from a preset spec."""
    return trial.suggest_int(
        name,
        spec["low"],
        spec["high"],
        step=spec.get("step", 1),
    )


def _int_grid_values(spec: dict[str, Any]) -> list[int]:
    """Expand an integer search spec into explicit grid values."""
    step = int(spec.get("step", 1))
    return list(range(int(spec["low"]), int(spec["high"]) + 1, step))


def _suggest_float_from_spec(
    trial: optuna.Trial,
    name: str,
    spec: dict[str, Any],
) -> float:
    """Sample a float parameter from a preset spec."""
    if spec["type"] == "fixed":
        return float(spec["value"])
    return trial.suggest_float(
        name,
        spec["low"],
        spec["high"],
        step=spec.get("step"),
    )


def _sample_chunking_params(
    trial: optuna.Trial,
    preset: dict[str, Any],
) -> dict[str, Any]:
    """Sample chunking parameters from either categorical or range presets."""
    if "chunk_size_choices" in preset:
        chunk_size = trial.suggest_categorical(
            "chunk_size",
            list(preset["chunk_size_choices"]),
        )
        if not isinstance(chunk_size, int):
            raise TypeError(f"chunk_size must be an int, got {chunk_size!r}")
    else:
        chunk_size = _suggest_int_from_spec(trial, "chunk_size", preset["chunk_size"])

    overlap_ratio = _suggest_float_from_spec(
        trial,
        "chunk_overlap_ratio",
        preset["chunk_overlap_ratio"],
    )
    return {
        "chunk_size": chunk_size,
        "chunk_overlap_ratio": overlap_ratio,
        "chunk_overlap": int(chunk_size * overlap_ratio),
    }


def _sample_retrieval_params(
    trial: optuna.Trial,
    preset: dict[str, Any],
) -> dict[str, Any]:
    """Sample retrieval candidate pools from either k-set or range presets."""
    if "k_set_choices" in preset:
        k_set = trial.suggest_categorical("k_set", list(preset["k_set_choices"]))
        k_values = preset["k_set_choices"][k_set]
        return {
            "k_set": k_set,
            "similarity_k": k_values["similarity_k"],
            "bm25_k": k_values["bm25_k"],
        }

    return {
        "similarity_k": _suggest_int_from_spec(
            trial,
            "similarity_k",
            preset["similarity_k"],
        ),
        "bm25_k": _suggest_int_from_spec(trial, "bm25_k", preset["bm25_k"]),
    }


def _sample_retrieval_range_params(
    trial: optuna.Trial,
    preset: dict[str, Any],
) -> dict[str, Any]:
    """Sample independent retrieval k ranges for retrieval-only tuning."""
    if "similarity_k" in preset and "bm25_k" in preset:
        return _sample_retrieval_params(trial, preset)

    return {
        "similarity_k": trial.suggest_int(
            "similarity_k",
            preset["retrieval_similarity_min"],
            preset["retrieval_similarity_max"],
        ),
        "bm25_k": trial.suggest_int(
            "bm25_k",
            preset["retrieval_bm25_min"],
            preset["retrieval_bm25_max"],
        ),
    }


def _sample_rrf_k(trial: optuna.Trial, preset: dict[str, Any]) -> int:
    """Return fixed RRF k or sample it from a TPE-friendly range."""
    if "rrf_k" in preset:
        return _suggest_int_from_spec(trial, "rrf_k", preset["rrf_k"])
    return preset["fixed_rrf_k"]


def _sample_final_k(trial: optuna.Trial, preset: dict[str, Any]) -> int:
    """Sample final_k from the preset specification."""
    return _suggest_int_from_spec(trial, "final_k", preset["final_k"])


def estimate_final_context_chars(params: dict[str, Any]) -> int:
    """Estimate sent-to-LLM evidence chars before evaluation."""
    final_k = params["final_k"]
    if final_k <= 0:
        return 0

    return final_k * params["chunk_size"]


def infer_search_family(search_space_preset: str) -> str:
    """Classify a preset into a search family based on its parameter specs."""
    preset = SEARCH_SPACE_PRESETS[search_space_preset]
    if "fixed_chunk_size" in preset:
        return "retrieval"
    if "k_set_choices" in preset or ("similarity_k" in preset and "bm25_k" in preset):
        return "joint"
    return "chunking"


def define_search_space(
    trial: optuna.Trial,
    *,
    search_space_preset: str,
) -> dict[str, Any]:
    """
    Define the hyperparameter search space for Optuna.

    Each parameter can be adjusted by changing the low/high bounds or
    adding/removing categorical choices.

    Args:
        trial: Optuna trial object for suggesting parameters.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    preset = SEARCH_SPACE_PRESETS[search_space_preset]
    search_family = infer_search_family(search_space_preset)
    params: dict[str, Any] = {}

    if search_family == "joint":
        params.update(_sample_chunking_params(trial, preset))
        params.update(_sample_retrieval_params(trial, preset))
        if "final_k" in preset:
            params["final_k"] = _sample_final_k(trial, preset)
    elif search_family == "chunking":
        # Fixed retrieval configuration for this search strategy. We keep the
        # retrieval candidate pools constant so Optuna only explores chunking size.
        params["similarity_k"] = preset["fixed_similarity_k"]
        params["bm25_k"] = preset["fixed_bm25_k"]

        # Keep overlap proportional to chunk size so larger chunks do not
        # automatically benefit from a separately tuned redundancy budget.
        params.update(_sample_chunking_params(trial, preset))
        if "final_k" in preset:
            params["final_k"] = _sample_final_k(trial, preset)
    elif search_family == "retrieval":
        # Fixed chunking configuration so Optuna only explores retrieval pools
        # and the final retrieval window size.
        params["chunk_size"] = preset["fixed_chunk_size"]
        params["chunk_overlap"] = preset["fixed_chunk_overlap"]
        params.update(_sample_retrieval_range_params(trial, preset))
        params["final_k"] = _sample_final_k(trial, preset)
    else:
        raise ValueError(
            f"Unsupported search family derived from preset: {search_space_preset}."
        )

    if "final_k" not in params:
        # Keep the final window fixed when chunking is part of the search so
        # chunk-size comparisons stay anchored to a stable downstream context.
        params["final_k"] = preset["fixed_final_k"]

    # Keep RRF fixed while we tune only the candidate pool sizes.
    params["rrf_k"] = _sample_rrf_k(trial, preset)

    # -------------------------------------------------------------------------
    # [ADJUST] Conditional/Dependent Parameters (Optional)
    # -------------------------------------------------------------------------
    # Example: Different parameters for different retrieval modes
    #
    # params["retrieval_mode"] = trial.suggest_categorical(
    #     "retrieval_mode", ["similarity", "bm25", "hybrid"]
    # )
    #
    # if params["retrieval_mode"] == "hybrid":
    #     # Hybrid mode: both similarity_k and bm25_k are active
    #     pass
    # elif params["retrieval_mode"] == "similarity":
    #     # Pure similarity: bm25_k is irrelevant
    #     params["bm25_k"] = 0
    # else:  # bm25 only
    #     params["similarity_k"] = 0

    return params


def get_sampler(
    search_method: str,
    *,
    seed: int,
    search_space_preset: str,
) -> optuna.samplers.BaseSampler:
    """Create the requested Optuna sampler for the selected search space."""
    normalized_method = search_method.lower()
    preset = SEARCH_SPACE_PRESETS[search_space_preset]
    search_family = infer_search_family(search_space_preset)

    if normalized_method == "tpe":
        if search_space_preset == "tpe_wide":
            return TPESampler(
                seed=seed,
                n_startup_trials=20,
                n_ei_candidates=64,
            )
        return TPESampler(seed=seed)

    if normalized_method == "grid":
        if search_family in {"joint", "chunking"} and "chunk_size_choices" not in preset:
            raise ValueError(
                f"Search space preset '{search_space_preset}' is range-based and "
                "is intended for TPE or random search, not grid search."
            )
        if search_family == "joint":
            search_space = {
                "chunk_size": list(preset["chunk_size_choices"]),
            }
            if "k_set_choices" in preset:
                search_space["k_set"] = list(preset["k_set_choices"])
            else:
                search_space["similarity_k"] = _int_grid_values(
                    preset["similarity_k"]
                )
                search_space["bm25_k"] = _int_grid_values(preset["bm25_k"])
            if "final_k" in preset:
                search_space["final_k"] = _int_grid_values(preset["final_k"])
        elif search_family == "chunking":
            search_space = {
                "chunk_size": list(preset["chunk_size_choices"]),
            }
            if "final_k" in preset:
                search_space["final_k"] = list(
                    range(
                        preset["final_k"]["low"],
                        preset["final_k"]["high"] + 1,
                        preset["final_k"].get("step", 1),
                    )
                )
        elif search_family == "retrieval":
            if "similarity_k" in preset and "bm25_k" in preset:
                search_space = {
                    "similarity_k": _int_grid_values(preset["similarity_k"]),
                    "bm25_k": _int_grid_values(preset["bm25_k"]),
                    "final_k": _int_grid_values(preset["final_k"]),
                }
            else:
                search_space = {
                    "similarity_k": list(
                        range(
                            preset["retrieval_similarity_min"],
                            preset["retrieval_similarity_max"] + 1,
                        )
                    ),
                    "bm25_k": list(
                        range(
                            preset["retrieval_bm25_min"],
                            preset["retrieval_bm25_max"] + 1,
                        )
                    ),
                    "final_k": _int_grid_values(preset["final_k"]),
                }
        else:
            raise ValueError(
                f"Unsupported search family derived from preset: {search_space_preset}."
            )
        return GridSampler(
            search_space=search_space
        )

    if normalized_method == "random":
        return RandomSampler(seed=seed)

    raise ValueError(
        f"Unsupported search method: {search_method}. "
        f"Expected one of {SEARCH_METHOD_CHOICES}."
    )


def get_pruner(search_method: str) -> optuna.pruners.BasePruner | None:
    """Return the pruner to use for the selected search method."""
    normalized_method = search_method.lower()

    if normalized_method in {"grid", "random"}:
        # Grid search and random search are both more useful here when every
        # sampled configuration runs to completion and gets a comparable score.
        return NopPruner()

    return MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=0,    # Prune immediately after first report
    )


# =============================================================================
# Benchmark Runner with Configurable Parameters
# =============================================================================

def run_benchmark_trial(
    params: dict[str, Any],
    dataset: list[dict[str, Any]],
    bm25_query_cache: dict[str, dict[str, Any]],
    embedding_model: Any,
    embedding_model_id: str,
    *,
    recall_ks: tuple[int, ...],
    hit_ks: tuple[int, ...],
    precision_ks: tuple[int, ...],
    chunking_strategy: ChunkingStrategy = DEFAULT_CHUNKING_STRATEGY,
    use_reranker: bool = False,
) -> dict[str, Any]:
    """
    Run a single benchmark trial with the given parameters.

    Args:
        params: Hyperparameter dictionary from define_search_space()
        dataset: Loaded benchmark dataset
        bm25_query_cache: Pre-computed BM25 query cache
        embedding_model: Initialized embedding model
        use_reranker: Whether to enable reranking during retrieval trials.

    Returns:
        Dictionary with evaluation metrics.
    """
    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    final_context_char_counts: list[int] = []

    # Extract parameters
    chunk_size = params["chunk_size"]
    chunk_overlap = params["chunk_overlap"]
    similarity_k = params["similarity_k"]
    bm25_k = params["bm25_k"]
    final_k = params["final_k"]
    rrf_k = params["rrf_k"]

    # Iterate through contracts
    for contract in dataset:
        title = contract["title"]
        context = contract["context"]
        qas = contract["qas"]

        # The cache key includes chunking config, so trials with the same
        # chunking can reuse embeddings while different chunk sizes stay isolated.
        embedded_database = get_or_create_persistent_vector_store(
            title=title,
            context=context,
            embedding_model=embedding_model,
            embedding_model_id=embedding_model_id,
            cache_root=DEFAULT_VECTOR_CACHE_DIR,
            chunking_config=chunker.ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
            chunking_strategy=chunking_strategy,
        )

        try:
            # Evaluate each question
            for qa in qas:
                question = qa["question"]
                qa_answers = qa["answers"]
                expected_answer_annotations = build_cuad_answer_annotations(qa_answers)
    
                # Build cache key
                cache_key = build_bm25_cache_key(
                    title=title,
                    question_id=qa["id"],
                    question=question,
                )
                cached_bm25 = bm25_query_cache.get(cache_key, {})
                bm25_query_override = cached_bm25.get("bm25_query")
    
                # Run retrieval through the shared benchmark graph path.
                started = time.perf_counter()
                answer_result = get_answer(
                    query=question,
                    embedded_database=embedded_database,
                    chat_model_id=CHAT_MODEL_ID,
                    system_prompt=TUNING_SYSTEM_PROMPT,
                    retrieval_mode=RETRIEVAL_MODE,
                    similarity_k=similarity_k,
                    bm25_k=bm25_k,
                    final_k=final_k,
                    rrf_k=rrf_k,
                    bm25_query_override=bm25_query_override,
                    run_mode="retrieval_only",
                    use_reranker=use_reranker,
                )
                retrieved_chunks = answer_result["retrieved_chunks"]
                final_retrieved_chunks = answer_result["final_retrieved_chunks"]
                final_context_char_counts.append(
                    sum(len(chunk.get("content", "") or "") for chunk in final_retrieved_chunks)
                )
    
                latency = calculate_latency(started)
                latencies.append(latency)
    
                # Evaluate all metrics in one pass
                metrics = evaluate_retrieval_question(
                    expected_answer_annotations,
                    retrieved_chunks,
                    final_k=final_k,
                    final_retrieved_chunks=final_retrieved_chunks,
                    recall_ks=recall_ks,
                    hit_ks=hit_ks,
                    precision_ks=precision_ks,
                )
    
                results.append({
                    "has_valid_gold_spans": metrics["has_valid_gold_spans"],
                    "coverage": metrics["coverage"],
                    "retrieval_metrics": metrics["retrieval_metrics"],
                    "failure_analysis": metrics["failure_analysis"],
                    "latency": latency,
                })
        finally:
            release_persistent_vector_store(embedded_database)
            gc.collect()
    aggregated_results = aggregate_retrieval_results(
        results,
        recall_ks=recall_ks,
        hit_ks=hit_ks,
        precision_ks=precision_ks,
    )

    return {
        **aggregated_results,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "max_latency": max(latencies) if latencies else 0,
        "avg_context_chars": (
            sum(final_context_char_counts) / len(final_context_char_counts)
            if final_context_char_counts
            else 0
        ),
    }


# =============================================================================
# [ADJUST] Objective Function - Modify scoring logic here
# =============================================================================

def create_objective(
    dataset: list[dict[str, Any]],
    bm25_query_cache: dict[str, dict[str, Any]],
    embedding_model: Any,
    embedding_model_id: str,
    search_space_preset: str,
    density_precision_target: float,
    chunking_strategy: ChunkingStrategy = DEFAULT_CHUNKING_STRATEGY,
    use_reranker: bool = False,
    budget_pruning_enabled: bool = True,
) -> Callable[[optuna.Trial], float]:
    """
    Create the Optuna objective function.

    [ADJUST] Modify the scoring logic below to prioritize different metrics:
    - Coverage rate (primary)
    - Context-size efficiency
    - Fully covered rate (strict accuracy)

    Args:
        dataset: Benchmark dataset
        bm25_query_cache: Pre-computed BM25 queries
        embedding_model: Initialized embedding model
        use_reranker: Whether to enable reranking during retrieval trials.
        budget_pruning_enabled: If True, skip configurations whose estimated
            final context exceeds the fixed context budget.
        density_precision_target: Precision value that maps to a density score
            of 1.0. Use 0.0 to disable the density contribution.

    Returns:
        Objective function for optuna.optimize()
    """

    preset = SEARCH_SPACE_PRESETS[search_space_preset]
    recall_ks = tuple(preset["recall_ks"])
    hit_ks = tuple(preset["hit_ks"])
    precision_ks = tuple(preset["precision_ks"])

    # Track evaluated parameter combinations to avoid duplicate trials
    _evaluated_cache: dict[tuple[tuple[str, Any], ...], float] = {}

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective for single-objective tuning."""
        params = define_search_space(
            trial,
            search_space_preset=search_space_preset,
        )
        estimated_context_chars = estimate_final_context_chars(params)
        trial.set_user_attr("params", params)
        trial.set_user_attr("estimated_context_chars", estimated_context_chars)
        trial.set_user_attr("context_char_budget", FINAL_CONTEXT_CHAR_BUDGET)
        trial.set_user_attr("budget_pruning_enabled", budget_pruning_enabled)

        if (
            budget_pruning_enabled
            and estimated_context_chars > FINAL_CONTEXT_CHAR_BUDGET
        ):
            # Skip benchmark execution entirely when the fixed final window is
            # guaranteed to exceed the context budget for this configuration.
            trial.set_user_attr("pruned_reason", "estimated_context_budget_exceeded")
            raise optuna.TrialPruned(
                "Estimated final context exceeds "
                f"{FINAL_CONTEXT_CHAR_BUDGET} chars: {estimated_context_chars}"
            )

        # Check if this parameter combination has already been evaluated
        param_key = _param_key(params)
        if param_key in _evaluated_cache:
            # Return cached result to avoid redundant computation
            cached_value = _evaluated_cache[param_key]
            trial.set_user_attr("deduplicated", True)
            trial.set_user_attr("original_params", dict(params))
            return cached_value

        try:
            # Run benchmark with these parameters
            metrics = run_benchmark_trial(
                params=params,
                dataset=dataset,
                bm25_query_cache=bm25_query_cache,
                embedding_model=embedding_model,
                embedding_model_id=embedding_model_id,
                recall_ks=recall_ks,
                hit_ks=hit_ks,
                precision_ks=precision_ks,
                chunking_strategy=chunking_strategy,
                use_reranker=use_reranker,
            )

            # -----------------------------------------------------------------
            # [ADJUST] Scoring Logic - Modify weights and thresholds as needed
            # -----------------------------------------------------------------

            partial_or_better_rate = metrics["coverage"]["partial_or_better"]["rate"]
            recall_at_final_k = metrics["metrics"]["recall"]["at_final_k_mean"] or 0.0
            hit_at_5 = metrics["metrics"]["hit"].get("at_5_mean") or 0.0
            fully_covered_at_final_k = metrics["metrics"]["context_quality"][
                "fully_covered_at_final_k_mean"
            ] or 0.0
            mrr_mean = metrics["metrics"]["ranking"]["mrr_mean"] or 0.0
            precision_at_final_k = (
                metrics["metrics"]["precision"]["at_final_k_mean"] or 0.0
            )
            mean_overlap_ratio_at_final_k = metrics["metrics"]["context_quality"][
                "mean_overlap_ratio_at_final_k_mean"
            ] or 0.0
            avg_latency = metrics["avg_latency"]
            max_latency = metrics["max_latency"]
            avg_context_chars = metrics["avg_context_chars"]

            density_score = (
                min(precision_at_final_k / density_precision_target, 1.0)
                if density_precision_target > 0
                else 0.0
            )

            quality_score = (
                0.30 * recall_at_final_k
                + 0.18 * partial_or_better_rate
                + 0.22 * fully_covered_at_final_k
                + 0.15 * mrr_mean
                + 0.10 * hit_at_5
                + 0.05 * mean_overlap_ratio_at_final_k
            )
            # Store metrics for analysis
            trial.set_user_attr("quality_score", quality_score)
            trial.set_user_attr("density_score", density_score)
            trial.set_user_attr("density_precision_target", density_precision_target)
            trial.set_user_attr("precision_at_final_k", precision_at_final_k)
            trial.set_user_attr("summary", metrics)
            trial.set_user_attr("metrics", metrics["metrics"])
            trial.set_user_attr("coverage", metrics["coverage"])
            trial.set_user_attr("failure_analysis", metrics["failure_analysis"])
            trial.set_user_attr("avg_latency", avg_latency)
            trial.set_user_attr("max_latency", max_latency)
            trial.set_user_attr("avg_context_chars", avg_context_chars)

            # Single objective: quality score only
            score = quality_score
            _evaluated_cache[param_key] = score

            # [DISABLED] Latency penalty - removed due to caching mechanism
            # With BM25 query cache and embedded vector reuse, latency measurements
            # no longer reflect true retrieval performance. First runs are slower
            # (cache generation), subsequent runs are artificially fast (cache hits),
            # making latency penalties unfair and misleading for optimization.

            # Report intermediate value for pruning
            trial.report(score, step=0)

            # -----------------------------------------------------------------
            # [ADJUST] Pruning - Early stopping for poor trials
            # -----------------------------------------------------------------
            if trial.should_prune():
                raise optuna.TrialPruned()

            return score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            # Mark infrastructure/runtime errors as failed trials so they do
            # not pollute the objective distribution as real zero-score configs.
            trial.set_user_attr("error", str(e))
            raise

    return objective


def _param_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Build a stable hashable key for a trial parameter dict."""
    return tuple(sorted(params.items()))


def _complete_trial_params(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
    """Return full evaluated params, including knobs fixed by the target."""
    params = dict(trial.user_attrs["params"])
    if "chunk_overlap" not in params:
        overlap_ratio = params["chunk_overlap_ratio"]
        params["chunk_overlap"] = int(params["chunk_size"] * overlap_ratio)
    required_keys = ("chunk_size", "similarity_k", "bm25_k", "final_k", "rrf_k")
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise KeyError(f"Trial params missing required keys: {missing_keys}")
    return params


def enqueue_zero_score_retries(study: optuna.Study) -> int:
    """
    Re-enqueue completed zero-score trials that have not already succeeded.

    This helps recover from transient runtime failures that were recorded as a
    0.0 objective value by the objective wrapper.
    """
    successful_keys = {
        _param_key(trial.params)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and trial.value > 0.0
    }
    queued_keys = set()
    retry_count = 0

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        if trial.value is None or trial.value != 0.0:
            continue
        if not trial.params:
            continue

        params_key = _param_key(trial.params)
        if params_key in successful_keys or params_key in queued_keys:
            continue

        study.enqueue_trial(dict(trial.params), skip_if_exists=False)
        queued_keys.add(params_key)
        retry_count += 1

    return retry_count


def _format_spec(name: str, spec: dict[str, Any]) -> str:
    """Return a compact CLI summary for a preset parameter spec."""
    if spec["type"] == "fixed":
        return f"{name}={spec['value']}"

    step = spec.get("step")
    step_text = f" step={step}" if step is not None else ""
    return f"{name}={spec['low']}-{spec['high']}{step_text}"


def _format_chunking_search(preset: dict[str, Any]) -> str:
    """Describe chunking dimensions for startup logging."""
    if "chunk_size_choices" in preset:
        chunk_size = f"chunk_size={list(preset['chunk_size_choices'])}"
    else:
        chunk_size = _format_spec("chunk_size", preset["chunk_size"])
    overlap = _format_spec("chunk_overlap_ratio", preset["chunk_overlap_ratio"])
    return f"{chunk_size}, {overlap}"


def _format_retrieval_search(preset: dict[str, Any]) -> str:
    """Describe retrieval dimensions for startup logging."""
    if "k_set_choices" in preset:
        return f"k_sets={preset['k_set_choices']}"
    return ", ".join(
        [
            _format_spec("similarity_k", preset["similarity_k"]),
            _format_spec("bm25_k", preset["bm25_k"]),
        ]
    )


def _format_retrieval_range_search(preset: dict[str, Any]) -> str:
    """Describe independent retrieval k ranges for retrieval-only tuning."""
    if "similarity_k" in preset and "bm25_k" in preset:
        return ", ".join(
            [
                _format_retrieval_search(preset),
                _format_spec("final_k", preset["final_k"]),
            ]
        )
    return (
        f"similarity_k={preset['retrieval_similarity_min']}-"
        f"{preset['retrieval_similarity_max']}, "
        f"bm25_k={preset['retrieval_bm25_min']}-{preset['retrieval_bm25_max']}, "
        f"{_format_spec('final_k', preset['final_k'])}"
    )


def _format_rrf_search(preset: dict[str, Any]) -> str:
    """Describe whether RRF k is fixed or sampled."""
    if "rrf_k" in preset:
        return _format_spec("rrf_k", preset["rrf_k"])
    return f"rrf_k={preset['fixed_rrf_k']}"


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point for config tuning."""
    started_at = datetime.now(timezone.utc)
    os.environ["LANGCHAIN_TRACING"] = "false"
    parser = argparse.ArgumentParser(
        description="Optimize RAG retrieval parameters using Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run
  python -m benchmark.retrieval_config_tuning --n-trials 10

  # Full tuning with 200 trials
  python -m benchmark.retrieval_config_tuning --n-trials 200

  # Resume interrupted run
  python -m benchmark.retrieval_config_tuning --storage sqlite:///benchmark/tuning.db

        """,
    )

    # [ADJUST] Command line arguments
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials to run (default: 50)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE_URL,
        help=f"Optuna storage URL (default: {DEFAULT_STORAGE_URL})",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
        help="Name of the Optuna study (required)",
    )
    parser.add_argument(
        "--limit-contracts",
        type=int,
        default=DEFAULT_LIMIT_CONTRACTS,
        help=f"Number of contracts per trial (default: {DEFAULT_LIMIT_CONTRACTS})",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=DEFAULT_LIMIT_QUESTIONS_PER_CONTRACT,
        help=f"Questions per contract (default: {DEFAULT_LIMIT_QUESTIONS_PER_CONTRACT})",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the LegalBench-RAG tuning dataset (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--disable-budget-pruning",
        action="store_true",
        help=(
            "Evaluate configurations even when estimated final context exceeds "
            "the fixed context budget. Use this for no-budget grid baselines."
        ),
    )
    parser.add_argument(
        "--density-precision-target",
        type=float,
        help=(
            "Precision@final_k value that maps to density_score=1.0; "
            "use 0 to disable the density contribution "
            "(default: selected search-space preset value)."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs (default: 1, use 1 for single-threaded)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--embedding-model-id",
        type=str,
        default=EMBEDDING_MODEL_ID,
        help=f"Embedding model ID to use (default: {EMBEDDING_MODEL_ID})",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        choices=["legal", "langchain_basic"],
        default=DEFAULT_CHUNKING_STRATEGY,
        help=(
            "Chunking strategy to tune: legal or langchain_basic "
            f"(default: {DEFAULT_CHUNKING_STRATEGY})"
        ),
    )
    parser.add_argument(
        "--search_method",
        "--search-method",
        type=str,
        choices=SEARCH_METHOD_CHOICES,
        default="tpe",
        help="Search method to use: tpe, grid, or random (default: tpe)",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        choices=SEARCH_SPACE_PRESET_CHOICES,
        required=True,
        help="Search space preset to use.",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for best parameters JSON",
    )
    parser.add_argument(
        "--use-reranker",
        dest="use_reranker",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranker during tuning (default: disabled).",
    )
    parser.add_argument(
        "--retry-zero-trials",
        action="store_true",
        help="Re-enqueue completed zero-score trials from the same study before running.",
    )

    args = parser.parse_args()
    search_space = SEARCH_SPACE_PRESETS[args.search_space]
    search_family = infer_search_family(args.search_space)
    density_precision_target = (
        args.density_precision_target
        if args.density_precision_target is not None
        else float(search_space["density_precision_target"])
    )
    bm25_query_cache_k = int(search_space["bm25_query_cache_k"])
    recall_ks = tuple(search_space["recall_ks"])
    hit_ks = tuple(search_space["hit_ks"])
    precision_ks = tuple(search_space["precision_ks"])

    print("=" * 70)
    print("Retrieval Configuration Tuning with Optuna")
    print("=" * 70)
    print(f"Study name: {args.study_name}")
    print(f"Storage: {args.storage}")
    print(f"Trials: {args.n_trials}")
    print(f"Contracts per trial: {args.limit_contracts}")
    print(f"Use reranker: {args.use_reranker}")
    print(f"Search family: {search_family}")
    print(f"Search method: {args.search_method}")
    print(f"Search space: {args.search_space}")
    print(f"Density precision target: {density_precision_target}")
    print(f"Optuna early pruning enabled: {args.search_method.lower() == 'tpe'}")
    print(f"Budget pruning enabled: {not args.disable_budget_pruning}")
    print(f"Random seed: {args.seed}")
    print(f"Embedding model: {args.embedding_model_id}")
    print(f"Chunking strategy: {args.chunking_strategy}")
    if search_family == "joint":
        fixed_parts = [_format_rrf_search(search_space)]
        if "final_k" not in search_space:
            fixed_parts.insert(0, f"final_k={search_space['fixed_final_k']}")
        print("Fixed config: " + ", ".join(fixed_parts))
        combined_parts = [
            _format_chunking_search(search_space),
            _format_retrieval_search(search_space),
        ]
        if "final_k" in search_space:
            combined_parts.append(_format_spec("final_k", search_space["final_k"]))
        print(
            "Combined search: "
            f"{', '.join(combined_parts)}"
        )
    elif search_family == "chunking":
        if "final_k" in search_space:
            print(
                "Fixed config: "
                f"similarity_k={search_space['fixed_similarity_k']}, "
                f"bm25_k={search_space['fixed_bm25_k']}, "
                f"{_format_rrf_search(search_space)}"
            )
            print(
                "Chunking search: "
                f"{_format_chunking_search(search_space)}, "
                f"{_format_spec('final_k', search_space['final_k'])}"
            )
        else:
            print(
                "Fixed config: "
                f"similarity_k={search_space['fixed_similarity_k']}, "
                f"bm25_k={search_space['fixed_bm25_k']}, "
                f"final_k={search_space['fixed_final_k']}, "
                f"{_format_rrf_search(search_space)}"
            )
            print(
                "Chunking search: "
                f"{_format_chunking_search(search_space)}"
            )
    else:
        print(
            "Fixed config: "
            f"chunk_size={search_space['fixed_chunk_size']}, "
            f"chunk_overlap={search_space['fixed_chunk_overlap']}, "
            f"{_format_rrf_search(search_space)}"
        )
        print(
            "Retrieval search: "
            f"{_format_retrieval_range_search(search_space)}"
        )
        if args.disable_budget_pruning:
            print(
                f"Context budget: {FINAL_CONTEXT_CHAR_BUDGET} chars "
                "(recorded only; pruning disabled)"
            )
        else:
            print(
                f"Context budget: {FINAL_CONTEXT_CHAR_BUDGET} chars "
                "(estimated pre-check)"
            )
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load dataset and prepare cache
    # -------------------------------------------------------------------------
    print("\nLoading dataset...")
    dataset = load_legalbench_tuning_dataset(
        args.dataset,
        limit_contracts=args.limit_contracts,
        limit_questions_per_contract=args.limit_questions,
    )
    print(f"Loaded {len(dataset)} contracts")

    # Check model connections
    print("\nChecking model connections...")
    if not pipeline.check_embedding_model_connection(args.embedding_model_id):
        print("ERROR: Embedding model connection failed")
        sys.exit(1)
    if RETRIEVAL_MODE != "hybrid":
        if not check_chat_model_connection(CHAT_MODEL_ID):
            print("ERROR: Chat model connection failed")
            sys.exit(1)

    embedding_model = pipeline.get_embeddings_model(args.embedding_model_id)

    # Load/prepare BM25 query cache
    print("\nPreparing BM25 query cache...")
    bm25_query_cache: dict[str, dict[str, Any]] = {}
    bm25_query_cache_path: Path | None = None
    if RETRIEVAL_MODE in {"bm25", "hybrid"}:
        dataset_stem = args.dataset.stem
        bm25_query_cache_path = (
            DEFAULT_BM25_QUERY_CACHE_DIR
            / f"bm25_query_cache_{dataset_stem}_{CHAT_MODEL_ID}_k{bm25_query_cache_k}.json"
        )
        bm25_query_cache = precompute_bm25_queries_if_needed(
            dataset=dataset,
            llm_mode=BM25_QUERY_LLM_MODE,
            chat_model_id=CHAT_MODEL_ID,
            cache_path=bm25_query_cache_path,
        )
        print(
            f"BM25 query cache path: {bm25_query_cache_path} "
            f"(cached questions: {len(bm25_query_cache)})"
        )
    else:
        print("Skipping BM25 query cache because retrieval mode does not use BM25.")

    # -------------------------------------------------------------------------
    # Create and run Optuna study
    # -------------------------------------------------------------------------

    # [ADJUST] Sampler and Pruner configuration
    sampler = get_sampler(
        args.search_method,
        seed=args.seed,
        search_space_preset=args.search_space,
    )
    pruner = get_pruner(args.search_method)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    # Create objective function
    objective = create_objective(
        dataset=dataset,
        bm25_query_cache=bm25_query_cache,
        embedding_model=embedding_model,
        embedding_model_id=args.embedding_model_id,
        use_reranker=args.use_reranker,
        search_space_preset=args.search_space,
        density_precision_target=density_precision_target,
        chunking_strategy=args.chunking_strategy,
        budget_pruning_enabled=not args.disable_budget_pruning,
    )

    if args.retry_zero_trials:
        retry_count = enqueue_zero_score_retries(study)
        print(f"Queued {retry_count} zero-score trial retries.")

    # Run optimization
    print(f"\nStarting optimization ({args.n_trials} trials)...")
    print("Press Ctrl+C to interrupt and save partial results\n")

    interrupted = False
    interruption_message: str | None = None

    try:
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs)
    except KeyboardInterrupt as exc:
        interrupted = True
        interruption_message = str(exc) or "KeyboardInterrupt"
        print("\n\nInterrupted by user. Saving results...")

    # -------------------------------------------------------------------------
    # Output results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("OPTIMIZATION RESULTS")
    print("=" * 70)

    best_trial = study.best_trial
    print(f"\nBest trial number: {best_trial.number}")
    print(f"Best score: {best_trial.value:.4f}")
    print(f"\nBest parameters:")
    for key, val in best_trial.params.items():
        print(f"  {key}: {val}")

    # Print additional metrics
    print(f"\nDetailed metrics:")
    for key in [
        "quality_score",
        "density_score",
        "precision_at_final_k",
        "avg_latency",
    ]:
        if key in best_trial.user_attrs:
            print(f"  {key}: {best_trial.user_attrs[key]:.4f}")
    print(
        "  partial_or_better_rate: "
        f"{best_trial.user_attrs['coverage']['partial_or_better']['rate']:.4f}"
    )
    print(
        "  fully_covered_rate: "
        f"{best_trial.user_attrs['coverage']['fully_covered']['rate']:.4f}"
    )
    print(
        "  recall_at_final_k: "
        f"{best_trial.user_attrs['metrics']['recall']['at_final_k_mean']:.4f}"
    )
    print(
        "  hit_at_5: "
        f"{best_trial.user_attrs['metrics']['hit']['at_5_mean']:.4f}"
    )

    # Save results to file
    output_path = args.output or DEFAULT_OUTPUT_DIR / f"best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_payload: dict[str, Any] = {}

    best_metrics: dict[str, Any] = {}
    best_params: dict[str, Any] = {}
    best_trial_number: int | None = None
    best_score: float | None = None

    best_trial = study.best_trial
    best_trial_number = best_trial.number
    best_score = study.best_value
    best_params = _complete_trial_params(best_trial)
    best_metrics = best_trial.user_attrs["summary"]
    summary_status = "interrupted" if interrupted else "completed"

    output_payload["best_trial"] = {
        "trial_number": study.best_trial.number,
        "score": study.best_value,
        "params": best_params,
        "metrics": study.best_trial.user_attrs["metrics"],
        "coverage": study.best_trial.user_attrs["coverage"],
        "failure_analysis": study.best_trial.user_attrs["failure_analysis"],
        "quality_score": study.best_trial.user_attrs["quality_score"],
        "density_score": study.best_trial.user_attrs["density_score"],
        "density_precision_target": study.best_trial.user_attrs[
            "density_precision_target"
        ],
        "precision_at_final_k": study.best_trial.user_attrs[
            "precision_at_final_k"
        ],
        "avg_latency": study.best_trial.user_attrs["avg_latency"],
        "max_latency": study.best_trial.user_attrs["max_latency"],
    }

    total_questions = int(best_metrics["total_questions"])
    valid_gold_question_count = int(best_metrics["answerable_questions"])
    no_valid_gold_question_count = int(best_metrics["no_answer_questions"])

    summary = {
        "run": {
            "status": summary_status,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output_path": str(output_path),
        },
        "dataset": {
            "dataset_path": str(args.dataset),
            "contracts_evaluated": len(dataset),
            "questions_evaluated": total_questions,
            "answerable_questions": valid_gold_question_count,
            "no_answer_questions": no_valid_gold_question_count,
        },
        "config": {
            "embedding_model_id": args.embedding_model_id,
            "chat_model_id": CHAT_MODEL_ID,
            "retrieval_mode": RETRIEVAL_MODE,
            "chunking_strategy": args.chunking_strategy,
            "objective": {
                "density_precision_target": density_precision_target,
            },
            "retrieval_params": {
                "similarity_k": best_params["similarity_k"],
                "bm25_k": best_params["bm25_k"],
                "final_k": best_params["final_k"],
                "rrf_k": best_params["rrf_k"],
            },
            "metric_k_config": {
                "recall_ks": list(recall_ks),
                "hit_ks": list(hit_ks),
                "precision_ks": list(precision_ks),
            },
            "chunking": {
                "chunk_size": best_params["chunk_size"],
                "chunk_overlap": best_params["chunk_overlap"],
            },
            "query_rewrite": {
                "llm_mode": BM25_QUERY_LLM_MODE,
                "bm25_query_cache_k": bm25_query_cache_k,
                "bm25_query_cache_path": (
                    str(bm25_query_cache_path)
                    if RETRIEVAL_MODE in {"bm25", "hybrid"}
                    else None
                ),
            },
            "storage": {
                "output_dir": str(DEFAULT_OUTPUT_DIR),
                "vector_cache_dir": str(DEFAULT_VECTOR_CACHE_DIR),
            },
        },
        "tuning": {
            "search_family": search_family,
            "search_method": args.search_method,
            "search_space": args.search_space,
            "trials_requested": args.n_trials,
            "trials_completed": len(study.trials),
            "best_trial_number": best_trial_number,
            "best_score": best_score,
            "budget_pruning_enabled": not args.disable_budget_pruning,
            "context_char_budget": FINAL_CONTEXT_CHAR_BUDGET,
        },
        "metrics": {
            **best_metrics["metrics"],
            "latency": {
                "avg_seconds": float(best_metrics["avg_latency"]),
                "max_seconds": float(best_metrics["max_latency"]),
            },
            "context_cost": {
                "avg_context_chars": float(best_metrics["avg_context_chars"]),
            },
        },
        "coverage": best_metrics["coverage"],
        "failure_analysis": best_metrics["failure_analysis"],
    }
    if interruption_message is not None:
        summary["run"]["interruption_message"] = interruption_message

    output_payload["summary"] = summary

    atomic_write_json(output_path, output_payload)
    print(f"\nResults saved to: {output_path}")

    # Print top 5 trials for reference
    print("\n" + "=" * 70)
    print("TOP 5 TRIALS")
    print("=" * 70)

    reverse = study.direction == optuna.study.StudyDirection.MAXIMIZE
    complete_trials = [
        (float(trial.value), trial)
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
    ]
    complete_trials.sort(reverse=reverse)
    sorted_trials = [trial for _, trial in complete_trials]
    for i, trial in enumerate(sorted_trials[:5], 1):
        value = trial.value
        assert value is not None
        print(f"\n{i}. Trial {trial.number}: score={value:.4f}")
        for key, val in trial.params.items():
            print(f"   {key}={val}", end=" ")
        print()

    # -------------------------------------------------------------------------
    # [ADJUST] Optional: Print instructions for viewing results
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VIEW RESULTS")
    print("=" * 70)
    print(f"\nDatabase location: {args.storage}")
    print("\nTo view results, run:")
    print(f"  optuna-dashboard {args.storage}")

    print("\nDone!")


if __name__ == "__main__":
    main()
