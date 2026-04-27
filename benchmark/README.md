# Benchmark Module

This module provides a comprehensive benchmarking framework for evaluating retrieval performance on legal question-answering tasks using the LegalBench-RAG dataset format.

The benchmark wrappers execute the shared production `rag_graph.py` graph(LangSmith observation disabled). The benchmark runs the shared graph in `retrieval_only` mode, so it evaluates retrieval quality without invoking final answer generation.

## Overview

The benchmark system evaluates how well the RAG (Retrieval-Augmented Generation) pipeline retrieves relevant text chunks that contain answers to legal questions. It supports three retrieval modes:

- **similarity**: Vector-based semantic search using embeddings
- **bm25**: Keyword-based retrieval using BM25 algorithm
- **hybrid**: Combined similarity and BM25 retrieval with fusion

The benchmark **does not** use the reranker by default. Pass `--use-reranker` to enable it. Enabling the reranker may affect runtime and retrieval metrics.

## Architecture

```
benchmark/
├── benchmark.py                        # Main entry point for running benchmarks
├── generation_latency_benchmark.py     # Speed test to find max "acceptable latency" across chunk sizes
├── chunking_strategies.py              # Benchmark chunking strategy adapters
├── gemini_batch.py                     # Gemini Batch API utilities for BM25 query cache
├── benchmark_pipeline.py               # Pipeline utilities for vector store and answer retrieval
├── benchmark_utils.py                  # Utility functions for data processing and metrics
├── build_embedding_data.py             # Script for pre-building embedding caches
├── chroma_cache/                       # Persistent Chroma vector store cache
├── chroma_cache_langchain_basic/       # Persistent cache for plain LangChain chunking
├── query_cache/                        # BM25 query rewrite cache
└── corpus/                             # Root directory for corpus text files
```

## Main Components

### 1. benchmark.py

The primary entry point for running benchmark evaluations.

**Key Features:**

- Supports batch, standard, and concurrent LLM modes
- Precomputes and caches BM25 query rewrites via Gemini Batch API
- Measures grouped retrieval metrics against expected answer spans
- Calculates grouped retrieval, coverage, failure-analysis, and latency statistics
- Handles graceful interruption with partial result saving

**Core Functions:**


| Function                              | Description                                           |
| ------------------------------------- | ----------------------------------------------------- |
| `set_benchmark_goal()`                | Routes to different benchmark pipelines based on goal |
| `run_retrieval_benchmark()`           | Executes the full retrieval benchmark                 |
| `precompute_bm25_queries_if_needed()` | Caches BM25 query rewrites before benchmark execution |


### 2. generation_latency_benchmark.py

A **speed-test benchmark** that measures end-to-end generation latency across different chunk sizes. Its purpose is to help users find the maximum **"acceptable latency"** for their local model setup, which in turn helps decide the ideal **chunk budget**.

**How it works:**

1. For each configured `chunk_size`, the benchmark runs retrieval + generation on the same set of questions.
2. Retrieval produces a realistic context window (using the same `retrieval_mode`, `final_k`, etc. as production).
3. The prompt is built from the retrieved chunks and the benchmark streams the full LLM generation.
4. Latency is measured per question, including time-to-first-token (TTFT) and output throughput.

**Why this matters for chunk budget:**

- Larger chunks -> fewer chunks needed to cover `final_k`, but each chunk contains more text.
- The total prompt size (context + query + system prompt) directly affects generation latency.
- By comparing latency across chunk sizes, you can identify the point where larger chunks start to make responses feel too slow for your use case.

**Key Features:**

- Tests multiple chunk sizes in a single run (default: `1000, 1500, 2000, 3000, 4000`)
- Measures per-question generation latency, TTFT, and estimated tokens per second
- Supports all retrieval modes (`similarity`, `bm25`, `hybrid`) and chunking strategies (`legal`, `langchain_basic`)
- Includes optional warmup generations to stabilise local model performance before measurement
- Saves detailed per-question results and per-chunk-size summary statistics

**Core Functions:**


| Function                           | Description                                                              |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `run_generation_latency_benchmark()` | Runs retrieval + generation across chunk sizes and collects latency stats |
| `summarize_chunk_results()`          | Aggregates latency, throughput, and token-count statistics per chunk size |
| `_stream_generation()`               | Streams generation from the chat model and records timing metrics        |


**Example usage:**

```bash
venv/bin/python -m benchmark.generation_latency_benchmark \
  --chunk-sizes 1000,2000,3000,4000 \
  --chunk-overlap-ratio 0.2 \
  --retrieval-mode similarity \
  --final-k 10 \
  --dataset benchmark/legalbench_sample_cuad_q200_c20_min10_seed42.json \
  --chat-model-id qwen-3.5-4b-mlx \
  --embedding-model-id Qwen3-Embedding-0.6B-4bit-DWQ \
  --warmup-generations 1
```

**Output:**

Results are saved to `benchmark_results/generation_latency_benchmark/generation_latency_results_{timestamp}.json`:

```json
{
  "summary": {
    "run": { "status": "completed", ... },
    "dataset": { "contexts_evaluated": 20, "questions_per_chunk_size": 200 },
    "config": { "chunk_sizes": [1000, 2000, 3000, 4000], ... },
    "metrics_by_chunk_size": {
      "1000": {
        "count": 200,
        "avg_latency_seconds": 2.5,
        "median_latency_seconds": 2.3,
        "p90_latency_seconds": 3.8,
        "avg_estimated_prompt_tokens": 3500,
        "avg_estimated_output_tokens_per_second": 42.0,
        "avg_time_to_first_token_seconds": 0.8
      },
      "4000": {
        "count": 200,
        "avg_latency_seconds": 8.2,
        "median_latency_seconds": 7.9,
        "p90_latency_seconds": 11.5,
        "avg_estimated_prompt_tokens": 14500,
        "avg_estimated_output_tokens_per_second": 38.0,
        "avg_time_to_first_token_seconds": 2.1
      }
    }
  },
  "results": [
    {
      "chunk_size": 1000,
      "query": "What is the termination clause?",
      "generation_latency_seconds": 2.4,
      "time_to_first_token_seconds": 0.7,
      "estimated_prompt_tokens": 3400,
      "estimated_output_tokens": 45,
      "output_chars_per_second": 18.5
    }
  ]
}
```

Use the `metrics_by_chunk_size` table to pick the largest chunk size that still meets your latency budget.


### 3. benchmark_pipeline.py

Provides pipeline utilities for vector store management and answer retrieval.

**Core Functions:**


| Function                                  | Description                                                                                  |
| ----------------------------------------- | -------------------------------------------------------------------------------------------- |
| `get_or_create_persistent_vector_store()` | Creates or reuses cached Chroma vector stores                                                |
| `get_answer()`                            | Executes the shared RAG graph with benchmark-managed resources and returns retrieval outputs |


### 4. gemini_batch.py

Uses the Google Generative SDK (`google-genai`) Batch API to generate BM25 query cache entries.

**Core Function:**


| Function                                 | Description                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------------------- |
| `populate_bm25_cache_via_gemini_batch()` | Submits inlined batch requests, polls job status, and writes BM25 query cache |


### 5. benchmark_metrics.py

Retrieval metric definitions and aggregation helpers used by benchmark and tuning.

**Core Functions:**


| Function                        | Description                                                          |
| ------------------------------- | -------------------------------------------------------------------- |
| `evaluate_retrieval_question()` | Computes grouped per-question retrieval metrics and failure analysis |
| `aggregate_mean()`              | Aggregates per-question metric values while skipping `null`          |
| `aggregate_miss_diagnostics()`  | Aggregates miss classifications at question and span level           |


### 6. benchmark_utils.py

Utility functions for chunk annotation, benchmark I/O, and shared helpers.

**Core Functions:**


| Function               | Description                                                           |
| ---------------------- | --------------------------------------------------------------------- |
| `annotate_documents()` | Annotates chunks with context offsets using source-aligned chunk text |
| `atomic_write_json()`  | Safely writes results atomically                                      |
| `short_string()`       | Truncates a string to a maximum length with `...` suffix              |


## Configuration

### Default Paths

```python
DEFAULT_DATASET_PATH = "benchmark/legalbench_sample_q200_c20_min5_seed42.json"
DEFAULT_CORPUS_DIR   = "benchmark/corpus"
DEFAULT_OUTPUT_DIR   = "benchmark_results"
DEFAULT_VECTOR_CACHE_DIR     = "benchmark/chroma_cache"
DEFAULT_BASELINE_VECTOR_CACHE_DIR = "benchmark/chroma_cache_langchain_basic"
DEFAULT_BM25_QUERY_CACHE_DIR = "benchmark/query_cache"
```

### Model Configuration

```python
# Embedding model for vector retrieval
embedding_model_id = "Qwen3-Embedding-0.6B-4bit-DWQ-Online"

# LLM model for BM25 query rewrite (batch/concurrent modes)
chat_model_id = "qwen-3.5-9b-openrouter"
```

### Retrieval Parameters

```python
retrieval_mode = "hybrid"           # Options: "similarity", "bm25", "hybrid"
similarity_k = 8                    # Candidate pool size from similarity search
bm25_k = 35                         # Candidate pool size from BM25 search
final_k = 10                        # Final reranked context window for downstream use
rrf_k = 80                          # RRF smoothing constant
recall_ks = (20,)                   # Recall@k buckets over the candidate pool
hit_ks = (5,)                       # Hit@k buckets over the candidate pool
precision_ks = (5, 10, 20)          # Precision@k buckets over the candidate pool
```

### Chunking Strategies

The benchmark can run with two chunking strategies:

- `legal`: the production legal-aware chunker from `chunker.py`
- `langchain_basic`: a plain `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap)` baseline with no legal separators, prefix merging, or section/header injection

Use `--chunking-strategy` to compare them directly while keeping the retrieval pipeline the same.

The persistent vector cache is also separated by strategy:

- `legal` uses `benchmark/chroma_cache/`
- `langchain_basic` uses `benchmark/chroma_cache_langchain_basic/`

### Benchmark Scope

```python
limit_contracts = 1000              # Max contexts (documents) to evaluate
limit_questions_per_contract = 1000 # Max questions per context
```

### LLM Modes

```python
llm_mode = "concurrent"             # Options: "batch", "standard", "concurrent"
```

- **batch**: Pre-computes BM25 query rewrites using batch API (faster, reproducible)
- **standard**: Computes BM25 queries on-the-fly (slower, live API calls)
- **concurrent**: Uses the standard API path but executes requests concurrently where supported

## Dataset Format

The benchmark expects **LegalBench-RAG** format JSON:

```json
{
  "tests": [
    {
      "query": "What is the termination clause?",
      "snippets": [
        {
          "file_path": "privacy_qa/SomeContract.txt",
          "span": [244, 312]
        }
      ]
    }
  ]
}
```

Each test has:

- `query`: the question to retrieve evidence for
- `snippets`: one or more gold answer spans, each with a `file_path` (relative to `corpus_dir`) and a `span` `[start, end]` character offset pair

The corpus text files referenced by `file_path` must exist under `--corpus-dir` (default: `benchmark/corpus/`).

## Dataset Sampling

For retrieval benchmarking, question datasets should be sampled in a cache-friendly way. Purely random question sampling tends to scatter questions across too many documents, which weakens Chroma cache reuse and makes runs slower and noisier.

The preferred pattern is:

- sample only valid questions (those with at least one gold span)
- group them by corpus file / document
- keep multiple questions per document
- keep a fixed number of questions per document when possible

In practice, the most stable benchmark datasets so far have used a balanced layout such as:

- `20` contexts
- `10` questions per context
- `200` questions total

Use `benchmark/dataset_cut3.py` to generate these datasets.

### Recommended Balanced Sampling

Generate a `200`-question benchmark dataset with `20` contexts, guaranteeing at least `5` questions per context:

```bash
python3 benchmark/dataset_cut3.py \
  --input-path benchmark/legalbench-datasets \ # or filename.json path
  --target-questions 200 \
  --target-contexts 20 \
  --min-per-context 5 \
  --seed 42 \
  --output benchmark/legalbench_sample_q200_c20_min5_seed42.json
```

Generate a `200`-question MAUD benchmark dataset with `20` contexts, guaranteeing at least `5` questions per context:

```bash
python3 benchmark/dataset_cut3.py \
  --input-path benchmark/legalbench-datasets/maud.json \
  --target-questions 200 \
  --target-contexts 20 \
  --min-per-context 5 \
  --seed 42 \
  --output benchmark/legalbench_sample_maud_q200_c20_min5_seed42.json
```

Generate a smaller `20`-question smoke-test dataset with `2` contexts, guaranteeing at least `5` questions per context:

```bash
python3 benchmark/dataset_cut3.py \
  --input-path benchmark/legalbench-datasets \
  --target-questions 20 \
  --target-contexts 2 \
  --min-per-context 5 \
  --seed 48 \
  --output benchmark/legalbench_sample_q20_c2_min5_seed48.json
```

Notes:

- `--input-path` can point either to a single LegalBench-RAG `.json` file or to a directory; when a directory is provided, the script recursively loads all `.json` files under it.
- the total dataset size is controlled explicitly by `--target-questions`
- the number of source documents is controlled by `--target-contexts`
- `--min-per-context` defines the guaranteed floor per selected context, not an exact per-context count
- after the guaranteed floor is allocated, the remaining questions are sampled randomly from the leftover pool across the selected contexts
- this format is intentionally designed to improve benchmark cache reuse while preserving some within-context variation

### Why This Sampling Mode

Compared with fully random valid-question sampling, balanced per-context
sampling has two practical advantages:

- fewer distinct documents are loaded during a run, so vector-store reuse is much better
- the benchmark better measures retrieval quality instead of spending most of its
time rebuilding tiny one-question caches

This is the recommended sampling strategy for parameter comparison runs.

### Mixed Config + Validation Splits

Use `benchmark/build_mix_dataset.py` when you want paired mixed datasets for
config tuning and validation that do not share source documents.

The mixed split files used below are generated artifacts, not files that are
expected to already exist in the repository:

- `benchmark/legalbench_mixed_config_q200_c20_seed42.json`
- `benchmark/legalbench_mixed_validation_q200_c20_seed42.json`

Generate them first from the original `benchmark/legalbench-datasets/*.json`
inputs before running config tuning or mixed-split benchmark evaluation.

The script samples each input file twice with the same per-source quota:

- first for the `config` split
- then for the `validation` split from the remaining contexts

This keeps the command simple while ensuring the two outputs are disjoint by
`context` / `file_path`.

Example command for the current four-task mix:

```bash
venv/bin/python -m benchmark.build_mix_dataset \
  --input benchmark/legalbench-datasets/cuad.json \
          benchmark/legalbench-datasets/maud.json \
          benchmark/legalbench-datasets/contractnli.json \
          benchmark/legalbench-datasets/privacy_qa.json \
  --max-contexts 7 5 5 3 \
  --max-questions 70 50 50 30 \
  --min-per-context 10 \
  --output-config benchmark/legalbench_mixed_config_q200_c20_seed42.json \
  --output-validation benchmark/legalbench_mixed_validation_q200_c20_seed42.json
```

Notes:

- `config` and `validation` use the same per-source quotas
- the two outputs do not share contexts
- each source must have enough eligible contexts to satisfy both splits
- `--max-contexts` and `--max-questions` are required for this dual-output flow

## Retrieval Config Tuning

For Optuna-based retrieval config tuning, the current recommended default dataset is the **config split** of the mixed dataset:

- `benchmark/legalbench_mixed_config_q200_c20_seed42.json` (LegalBench-RAG mixed split format)

If this file is missing, return to the "Mixed Config + Validation Splits"
section above and generate both mixed split files with
`benchmark/build_mix_dataset.py` first.

This mixed dataset combines multiple LegalBench-RAG sources (CUAD, MAUD, ContractNLI, Privacy QA) and provides better task diversity than a single-source dataset. The config split is used for hyperparameter optimization, while the validation split (`benchmark/legalbench_mixed_validation_q200_c20_seed42.json`) can be used for final evaluation of the best configuration.

Recommended command for a fresh tuning study:

```bash
venv/bin/python -m benchmark.retrieval_config_tuning \
  --dataset benchmark/legalbench_mixed_config_q200_c20_seed42.json \
  --n-trials 50 \
  --study-name "mixed_q200_recall_balanced_{timestamp}"
```

### Validation Split Evaluation

After tuning completes, evaluate the best configuration on the **validation split** to get an unbiased estimate of retrieval quality:

```bash
venv/bin/python -m benchmark.benchmark \
  --dataset benchmark/legalbench_mixed_validation_q200_c20_seed42.json \
  --chunk-size <best_chunk_size> \
  --chunk-overlap <best_chunk_overlap> \
  --similarity-k <best_similarity_k> \
  --bm25-k <best_bm25_k> \
  --final-k <best_final_k> \
  --rrf-k <best_rrf_k> \
  --retrieval-mode hybrid
```

Replace the `<best_*>` placeholders with the hyperparameters from the best trial reported by the tuning run. The validation split uses the same per-source quotas but disjoint source documents, so it measures generalization across unseen contracts.

Current recommended single-objective `quality_score`:

```python
quality_score = (
    0.34 * recall_at_final_k
    + 0.22 * partial_or_better_rate
    + 0.18 * fully_covered_at_final_k
    + 0.10 * mrr_mean
    + 0.08 * hit_at_5
    + 0.08 * mean_overlap_ratio_at_final_k
)
```

Why this weighting is recommended:

- `recall_at_final_k` stays primary so tuning strongly penalizes missed answer spans
- `partial_or_better_rate` keeps question-level spread stable across contracts
- `fully_covered_at_final_k` preserves a strict legal completeness signal
- `mrr_mean` and `hit_at_5` reward configs that surface the right chunk earlier
- `mean_overlap_ratio_at_final_k` still gives partial credit on long spans instead of treating every non-full hit the same

## Usage

### Running the Benchmark

```bash
# Run with current defaults (LegalBench-RAG dataset)
venv/bin/python -m benchmark.benchmark

# Example with explicit chunking + retrieval settings
venv/bin/python -m benchmark.benchmark \
  --llm-mode concurrent \
  --chunking-strategy legal \
  --chunk-size 3000 \
  --chunk-overlap 500 \
  --similarity-k 6 \
  --bm25-k 24 \
  --final-k 10 \
  --recall-ks 10,20,30 \
  --hit-ks 1,3,5 \
  --precision-ks 5,10,20,30 \
  --limit-contracts 10 \
  --use-reranker
```

Run the benchmark on the balanced `200`-question LegalBench-RAG dataset:

```bash
venv/bin/python -m benchmark.benchmark \
  --llm-mode concurrent \
  --dataset benchmark/legalbench_sample_q200_c20_min5_seed42.json \
  --corpus-dir benchmark/corpus \
  --chunking-strategy legal \
  --retrieval-mode hybrid \
  --chunk-size 1850 \
  --chunk-overlap 370 \
  --similarity-k 8 \
  --bm25-k 32 \
  --final-k 10 \
  --rrf-k 96 \
  --recall-ks 10,20,30 \
  --hit-ks 1,3,5 \
  --precision-ks 5,10,20,30 \
  --limit-contracts 1000 \
  --limit-questions 1000
```

Run the same benchmark with the plain LangChain chunking baseline:

```bash
venv/bin/python -m benchmark.benchmark \
  --llm-mode concurrent \
  --dataset benchmark/legalbench_sample_q200_c20_min5_seed42.json \
  --corpus-dir benchmark/corpus \ # Change from legal to langchain_basic
  --chunking-strategy langchain_basic \
  --retrieval-mode hybrid \
  --chunk-size 1850 \
  --chunk-overlap 370 \
  --similarity-k 8 \
  --bm25-k 32 \
  --final-k 10 \
  --rrf-k 96 \
  --recall-ks 10,20,30 \
  --hit-ks 1,3,5 \
  --precision-ks 5,10,20,30 \
  --limit-contracts 1000 \
  --limit-questions 1000
```

By default, retrieved chunk `content` is shortened in the saved JSON output.
Use `--no-short-retrieved-chunk-content` if you want full chunk text in
`results[*].retrieved_chunks` and `results[*].final_retrieved_chunks`.

By default, the benchmark records:

- `Recall@10` and `Recall@20`
- `Hit@1`, `Hit@3`, and `Hit@5`
- `Precision@5`, `Precision@10`, and `Precision@20`
- `MRR`, `Fully Covered@FinalK`, `Mean Overlap Ratio@FinalK`, and coverage counts

These defaults are enough to populate the standard retrieval comparison table
used in this repo without any post-hoc metric recomputation.

Other datasets:

`benchmark/legalbench_sample_contractnli_q200_c20_min10_seed42.json`
`benchmark/legalbench_sample_cuad_q200_c20_min10_seed42.json`
`benchmark/legalbench_sample_privacy_qa_q194_c7_min10_seed42.json`

### Building Embedding Cache (Pre-computation)

To pre-build all vector stores before running benchmarks:

```python
venv/bin/python -m benchmark.build_embedding_data
```

This processes all documents and creates persistent Chroma caches, significantly speeding up subsequent benchmark runs.

## Output Format

Results are saved to `benchmark_results/benchmark_results_{timestamp}.json`:

```json
{
  "summary": {
    "run": {
      "status": "completed",
      "started_at": "2025-01-15T10:30:00+00:00",
      "completed_at": "2025-01-15T10:45:00+00:00",
      "output_path": "benchmark_results/benchmark_results_20250115_104500.json"
    },
    "dataset": {
      "dataset_path": "...",
      "corpus_dir": "benchmark/corpus",
      "contexts_evaluated": 20,
      "questions_evaluated": 200,
      "answerable_questions": 196,
      "no_answer_questions": 4
    },
    "config": {
      "embedding_model_id": "Qwen3-Embedding-0.6B-4bit-DWQ-Online",
      "chat_model_id": "qwen-3.5-9b-openrouter",
      "retrieval_mode": "hybrid",
      "use_reranker": false,
      "retrieval_params": {
        "similarity_k": 8,
        "bm25_k": 35,
        "final_k": 10,
        "rrf_k": 80
      },
      "metric_k_config": {
        "recall_ks": [10, 20],
        "hit_ks": [1, 3, 5],
        "precision_ks": [5, 10, 20]
      },
      "chunking": {
        "strategy": "legal",
        "chunk_size": 2000,
        "chunk_overlap": 500
      },
      "query_rewrite": {
        "llm_mode": "concurrent",
        "bm25_query_cache_path": "benchmark/query_cache/bm25_query_cache_all.json"
      },
      "storage": {
        "output_dir": "benchmark_results",
        "vector_cache_dir": "benchmark/chroma_cache"
      }
    },
    "metrics": {
      "recall": {
        "at_10_mean": 0.70,
        "at_20_mean": 0.72,
        "at_final_k_mean": 0.68
      },
      "hit": {
        "at_1_mean": 0.58,
        "at_3_mean": 0.69,
        "at_5_mean": 0.74
      },
      "precision": {
        "at_5_mean": 0.48,
        "at_10_mean": 0.44,
        "at_20_mean": 0.43,
        "at_final_k_mean": 0.42
      },
      "f1": {
        "at_10_mean": 0.54,
        "at_20_mean": 0.55,
        "at_final_k_mean": 0.53
      },
      "ranking": {
        "mrr_mean": 0.61
      },
      "context_quality": {
        "fully_covered_at_final_k_mean": 0.63,
        "mean_overlap_ratio_at_final_k_mean": 0.81
      },
      "latency": {
        "avg_seconds": 1.23
      }
    },
    "coverage": {
      "fully_covered": {
        "count": 80,
        "rate": 0.80
      },
      "partial_covered": {
        "count": 10,
        "rate": 0.10
      },
      "not_covered": {
        "count": 10,
        "rate": 0.10
      },
      "partial_or_better": {
        "count": 90,
        "rate": 0.90
      }
    },
    "failure_analysis": {
      "diagnosed_miss_count": 20,
      "question_level": {
        "boundary_split": {
          "count": 6,
          "rate": 0.30
        },
        "true_miss": {
          "count": 14,
          "rate": 0.70
        },
        "mixed_miss": {
          "count": 2,
          "rate": 0.10
        }
      },
      "span_level": {
        "boundary_split": {
          "count": 9,
          "rate": 0.36
        },
        "true_miss": {
          "count": 16,
          "rate": 0.64
        },
        "total_missing_spans": 25
      }
    }
  },
  "results": [
    {
      "file_path": "privacy_qa/SomeContract.txt",
      "question_id": "abc123def456",
      "query": "What is...?",
      "bm25_query": "optimized search terms",
      "bm25_query_source": "cache",
      "bm25_query_rewrite_applied": true,
      "bm25_query_rewrite_error": null,
      "answer_annotations": [
        {"answer_start": 244, "answer_end": 312}
      ],
      "retrieved_chunks": [...],
      "final_retrieved_chunks": [...],
      "coverage": "Yes",
      "has_valid_gold_spans": true,
      "retrieval_metrics": {
        "recall": {
          "at_10": 1.0,
          "at_20": 1.0,
          "at_final_k": 1.0
        },
        "hit": {
          "at_1": 1.0,
          "at_3": 1.0,
          "at_5": 1.0
        },
        "precision": {
          "at_5": 0.52,
          "at_10": 0.44,
          "at_20": 0.43,
          "at_final_k": 0.42
        },
        "f1": {
          "at_10": 0.58,
          "at_20": 0.59,
          "at_final_k": 0.57
        },
        "ranking": {
          "mrr": 1.0
        },
        "context_quality": {
          "fully_covered_at_final_k": 1.0,
          "mean_overlap_ratio_at_final_k": 1.0
        }
      },
      "failure_analysis": {
        "classification": "fully_covered",
        "missing_span_count": 0,
        "boundary_split_span_count": 0,
        "true_miss_span_count": 0
      },
      "latency_seconds": 1.5
    }
  ]
}
```

## Metrics

### Answer Coverage

The primary retrieval quality metric:


| Classification | Description                                                                     |
| -------------- | ------------------------------------------------------------------------------- |
| **Yes**        | All annotated gold spans are contained in the final context window              |
| **partial**    | Some (but not all) annotated gold spans are covered in the final context window |
| **No**         | None of the annotated gold spans are covered in the final context window        |
| **N/A**        | The question has no valid gold span annotations, so coverage is not evaluated   |


For questions that are not fully covered, the benchmark also reports a miss diagnosis in `failure_analysis`:

- `boundary_split`: overlapping retrieved chunks together cover the gold span, so the miss is likely caused by chunk boundaries.
- `true_miss`: retrieval still misses part or all of the gold span even after considering every overlapping retrieved chunk.
- `mixed_miss`: the same question contains both boundary-split spans and true-miss spans.
- `*_count`: question-level counts.
- `*_span_count`: gold-span-level counts, useful when one question has multiple annotated answers.

### Recall Metrics

Recall metrics are stored under `results[*].retrieval_metrics.recall` and aggregated into `summary.metrics.recall`.

For any configured `k` value:

`Recall@k = covered valid gold spans in the top k candidate chunks / total valid gold spans`

- A gold span is valid when `answer_start >= 0` and `answer_end > answer_start`
- A span counts as covered when at least one retrieved chunk fully contains that span
- If a question has no valid gold spans, recall values are stored as `null` and excluded from summary means
- Candidate-pool metrics are computed from `results[*].retrieved_chunks`
- If fewer than `k` candidate chunks are available, the metric is computed over the available ranked chunks only

The benchmark also always records `at_final_k`, which is computed from `results[*].final_retrieved_chunks` and aligned with the final context window sent downstream.

### Hit Metrics

Hit metrics are stored under `results[*].retrieval_metrics.hit` and aggregated into `summary.metrics.hit`.

For any configured `k` value:

`Hit@k = 1 if any valid gold span is covered in the top k candidate chunks else 0`

- Stored per question in `results[*].retrieval_metrics.hit.at_{k}`
- Aggregated into `summary.metrics.hit.at_{k}_mean`
- Useful for distinguishing configs that surface relevant evidence earlier

### Precision Metrics

Precision metrics are stored under `results[*].retrieval_metrics.precision` and aggregated into `summary.metrics.precision`.

For any configured `k` value:

`Precision@k = relevant retrieved chunks / retrieved chunks in the top k candidate chunks`

- A retrieved chunk is relevant when it fully covers at least one valid gold span
- Candidate-pool precision metrics are computed from `results[*].retrieved_chunks`
- The benchmark also always records `at_final_k`, computed from `results[*].final_retrieved_chunks`

### F1 Metrics

F1 metrics are stored under `results[*].retrieval_metrics.f1` and aggregated into `summary.metrics.f1`.

For any `k` value that appears in both `recall_ks` and `precision_ks`:

`F1@k = 2 * (Recall@k * Precision@k) / (Recall@k + Precision@k)`

- Computed as the harmonic mean of recall and precision
- Only calculated for k values shared between the recall and precision configurations
- `at_final_k` is always included
- Returns `null` when either recall or precision is `null`

### Fully Covered@FinalK

Binary strict-coverage metric over the final context window:

`FullyCovered@FinalK = 1 if all valid gold spans are covered in top final_k else 0`

- Stored per question in `results[*].retrieval_metrics.context_quality.fully_covered_at_final_k` and averaged into `summary.metrics.context_quality.fully_covered_at_final_k_mean`
- Equivalent in scope to the global `coverage` summary, which is also defined over the final context window

### Mean Overlap Ratio@FinalK

Partial-credit span overlap metric over the final context window:

For each gold span, compute the best overlap ratio against the top `final_k` chunks, then average over spans.

- Returns `1.0` when every gold span is fully covered
- Returns a fractional value when chunks only partially overlap a gold span
- Especially useful when tuning `chunk_size` and `chunk_overlap`

### MRR

Mean Reciprocal Rank of the first relevant chunk:

`MRR = 1 / rank_of_first_chunk_that_covers_any_gold_span`

- Stored per question as `results[*].retrieval_metrics.ranking.mrr`
- Aggregated as `summary.metrics.ranking.mrr_mean`

### Latency

Per-question execution time measured from query submission to retrieval completion.

## Caching Strategy

### Vector Store Cache

- Location:
`benchmark/chroma_cache/{content_hash}_{model_hash}_{cfg_hash}/`
- The chunking strategy is part of the cache key, so `legal` and
`langchain_basic` do not share persisted vector stores
- `content_hash` is derived from the corpus file text, so the same source document
can reuse the same cache across different benchmark datasets
- Persists embedded chunks per corpus file
- Reused across benchmark runs with same embedding model and chunking config
- Significantly reduces embedding computation time
- The plain baseline uses a separate cache root:
`benchmark/chroma_cache_langchain_basic/`

### BM25 Query Cache

- Current location: `benchmark/query_cache/bm25_query_cache_all.json`
- Stores rewritten BM25 queries keyed by `(file_path, question_id, query)`
- In `batch` mode, missing entries are generated with Gemini Batch API (`google-genai`)
- Batch requests are sent as `inlined_requests`; response mapping uses request metadata (`cache_key`)
- In `standard` mode, cache is read but missing entries are not pre-generated
- In `concurrent` mode, the same cache is reused while retrieval requests are executed concurrently where supported
- Enables fair comparison across retrieval configurations

## Environment Variables

```bash
# LangSmith tracing (optional)
LANGCHAIN_TRACING=false  # Default: disabled for benchmarks

# Required for Gemini Batch API when llm_mode=batch
GOOGLE_API_KEY=your_api_key
```

## Extending the Benchmark

### Adding a New Retrieval Mode

1. Update `retrieval_mode` validation in `run_retrieval_benchmark()`
2. Implement the retrieval logic in the shared `rag_graph.py` execution path
3. Update metrics calculation if needed

### Adding New Metrics

1. Add the metric in `benchmark_metrics.py`
2. Decide which metric group it belongs to (`recall`, `hit`, `precision`, `ranking`, or `context_quality`)
3. Update summary aggregation in `benchmark.py`
4. Update the output JSON example in this README

### Supporting New Dataset Formats

1. Modify `run_retrieval_benchmark()` to parse the new format
2. Ensure each test has a `query` and a `snippets` list with `file_path` and `span`
3. Place corresponding corpus text files under `corpus_dir`
