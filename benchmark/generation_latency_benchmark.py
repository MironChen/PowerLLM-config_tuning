from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from benchmark.benchmark_pipeline import get_or_create_persistent_vector_store
from benchmark.benchmark_utils import annotate_documents, atomic_write_json
from benchmark.chunking_strategies import (
    DEFAULT_CHUNKING_STRATEGY,
    ChunkingStrategy,
    build_benchmark_chunks,
)
from powerllm.models.model_resolver import check_chat_model_connection, get_chat_model
import powerllm.retrieval.chunker as chunker
import powerllm.retrieval.pipeline as pipeline
from powerllm.retrieval.rag_graph import (
    SYSTEM_PROMPT,
    _llm_content_to_text,
    build_generation_prompt_node,
    format_context_node,
    run_rag_query_with_resources,
)


DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parent / "legalbench_sample_cuad_q200_c20_min10_seed42.json"
)
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
DEFAULT_OUTPUT_DIR = Path("benchmark_results/generation_latency_benchmark")
DEFAULT_VECTOR_CACHE_DIR = Path("benchmark/chroma_cache")
DEFAULT_BASELINE_VECTOR_CACHE_DIR = Path("benchmark/chroma_cache_langchain_basic")
DEFAULT_CHUNK_SIZES = (1000, 1500, 2000, 3000, 4000)
DEFAULT_CHAT_MODEL_ID = "qwen-3.5-4b-mlx"
DEFAULT_EMBEDDING_MODEL_ID = "Qwen3-Embedding-0.6B-4bit-DWQ"
APPROX_CHARS_PER_TOKEN = 4


def parse_chunk_sizes(raw_value: str) -> tuple[int, ...]:
    values: list[int] = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid chunk size '{stripped}'. Expected comma-separated integers."
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"Invalid chunk size '{value}'. Chunk sizes must be positive integers."
            )
        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError("Expected at least one positive chunk size.")

    return tuple(dict.fromkeys(values))


def resolve_chunk_overlap(
    *,
    chunk_size: int,
    chunk_overlap: int | None,
    chunk_overlap_ratio: float,
) -> int:
    if chunk_overlap is not None:
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative.")
        return chunk_overlap
    if chunk_overlap_ratio < 0:
        raise ValueError("chunk_overlap_ratio must be non-negative.")
    return int(chunk_size * chunk_overlap_ratio)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    if percentile_value <= 0:
        return min(values)
    if percentile_value >= 100:
        return max(values)

    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * (percentile_value / 100)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1 - weight)
        + sorted_values[upper_index] * weight
    )


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / APPROX_CHARS_PER_TOKEN))


def safe_rate(
    *,
    numerator: int | float,
    denominator_seconds: int | float | None,
) -> float:
    if not isinstance(denominator_seconds, (int, float)) or denominator_seconds <= 0:
        return 0.0
    return float(numerator) / denominator_seconds


def log_progress(message: str) -> None:
    print(message, flush=True)


def summarize_chunk_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        item["generation_latency_seconds"]
        for item in results
        if isinstance(item.get("generation_latency_seconds"), (int, float))
    ]
    prompt_counts = [
        item["prompt_char_count"]
        for item in results
        if isinstance(item.get("prompt_char_count"), int)
    ]
    context_counts = [
        item["context_char_count"]
        for item in results
        if isinstance(item.get("context_char_count"), int)
    ]
    output_counts = [
        item["output_char_count"]
        for item in results
        if isinstance(item.get("output_char_count"), int)
    ]
    prompt_token_counts = [
        item["estimated_prompt_tokens"]
        for item in results
        if isinstance(item.get("estimated_prompt_tokens"), int)
    ]
    output_token_counts = [
        item["estimated_output_tokens"]
        for item in results
        if isinstance(item.get("estimated_output_tokens"), int)
    ]
    output_char_rates = [
        item["output_chars_per_second"]
        for item in results
        if isinstance(item.get("output_chars_per_second"), (int, float))
    ]
    output_token_rates = [
        item["estimated_output_tokens_per_second"]
        for item in results
        if isinstance(item.get("estimated_output_tokens_per_second"), (int, float))
    ]
    time_to_first_token_values = [
        item["time_to_first_token_seconds"]
        for item in results
        if isinstance(item.get("time_to_first_token_seconds"), (int, float))
    ]
    error_count = sum(1 for item in results if item.get("generation_error"))

    return {
        "count": len(results),
        "avg_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        "median_latency_seconds": statistics.median(latencies) if latencies else 0.0,
        "min_latency_seconds": min(latencies) if latencies else 0.0,
        "max_latency_seconds": max(latencies) if latencies else 0.0,
        "p50_latency_seconds": percentile(latencies, 50),
        "p90_latency_seconds": percentile(latencies, 90),
        "p95_latency_seconds": percentile(latencies, 95),
        "avg_prompt_char_count": (
            statistics.mean(prompt_counts) if prompt_counts else 0.0
        ),
        "avg_context_char_count": (
            statistics.mean(context_counts) if context_counts else 0.0
        ),
        "avg_output_char_count": (
            statistics.mean(output_counts) if output_counts else 0.0
        ),
        "avg_estimated_prompt_tokens": (
            statistics.mean(prompt_token_counts) if prompt_token_counts else 0.0
        ),
        "avg_estimated_output_tokens": (
            statistics.mean(output_token_counts) if output_token_counts else 0.0
        ),
        "avg_output_chars_per_second": (
            statistics.mean(output_char_rates) if output_char_rates else 0.0
        ),
        "avg_estimated_output_tokens_per_second": (
            statistics.mean(output_token_rates) if output_token_rates else 0.0
        ),
        "avg_time_to_first_token_seconds": (
            statistics.mean(time_to_first_token_values)
            if time_to_first_token_values
            else 0.0
        ),
        "p50_time_to_first_token_seconds": percentile(
            time_to_first_token_values,
            50,
        ),
        "p90_time_to_first_token_seconds": percentile(
            time_to_first_token_values,
            90,
        ),
        "total_generation_time_seconds": sum(latencies),
        "error_count": error_count,
    }


def _load_grouped_contexts(
    *,
    dataset_path: Path,
    limit_contracts: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tests: list[dict[str, Any]] = data["tests"]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for test in tests:
        fp = test["snippets"][0]["file_path"]
        grouped[fp].append(test)

    contexts = list(grouped.items())
    if limit_contracts < len(contexts):
        contexts = contexts[:limit_contracts]
    return contexts


def _requires_vector_retrieval(retrieval_mode: str) -> bool:
    return retrieval_mode in {"similarity", "hybrid"}


def _build_embedded_database(
    *,
    file_path: str,
    corpus_text: str,
    retrieval_mode: str,
    embedding_model: Any | None,
    embedding_model_id: str,
    vector_cache_dir: Path,
    chunking_config: chunker.ChunkingConfig,
    chunking_strategy: ChunkingStrategy,
) -> Any:
    if _requires_vector_retrieval(retrieval_mode):
        if embedding_model is None:
            raise ValueError(
                "An embedding model is required for similarity or hybrid retrieval."
            )
        return get_or_create_persistent_vector_store(
            title=file_path,
            context=corpus_text,
            embedding_model=embedding_model,
            embedding_model_id=embedding_model_id,
            cache_root=vector_cache_dir,
            chunking_config=chunking_config,
            chunking_strategy=chunking_strategy,
        )

    docs = build_benchmark_chunks(
        [
            Document(
                page_content=corpus_text,
                metadata={"title": file_path, "source": file_path},
            )
        ],
        strategy=chunking_strategy,
        chunking_config=chunking_config,
    )
    return annotate_documents(context=corpus_text, documents=docs)


def _run_retrieval_only(
    *,
    query: str,
    embedded_database: Any,
    chat_model_id: str,
    retrieval_mode: str,
    similarity_k: int,
    bm25_k: int,
    final_k: int,
    rrf_k: int,
    use_reranker: bool,
) -> dict[str, Any]:
    return run_rag_query_with_resources(
        case_id="benchmark",
        query=query,
        embedding_model_id="benchmark",
        chat_model_id=chat_model_id,
        vector_store=(
            embedded_database.get("vector_store")
            if isinstance(embedded_database, dict)
            else None
        ),
        documents=(
            embedded_database.get("documents")
            if isinstance(embedded_database, dict)
            else embedded_database
        ),
        retrieval_mode=retrieval_mode,
        similarity_k=similarity_k,
        bm25_k=bm25_k,
        final_k=final_k,
        rrf_k=rrf_k,
        system_prompt=SYSTEM_PROMPT,
        run_mode="retrieval_only",
        use_reranker=use_reranker,
    )


def _build_generation_prompt(
    *,
    query: str,
    retrieved_docs: list[Document],
) -> tuple[str, str]:
    context_text = format_context_node({"retrieved_docs": retrieved_docs})["context_text"]
    prompt = build_generation_prompt_node(
        {"query": query, "context_text": context_text}
    )["generation_prompt"]
    return prompt, context_text


def _stream_generation(
    *,
    model: Any,
    system_prompt: str,
    generation_prompt: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    time_to_first_token: float | None = None
    try:
        response_chunks: list[str] = []
        merged_chunk: Any | None = None
        log_progress("Starting generation stream...")
        for chunk in model.stream(
            [
                ("system", system_prompt),
                ("human", generation_prompt),
            ]
        ):
            merged_chunk = chunk if merged_chunk is None else merged_chunk + chunk
            text = _llm_content_to_text(getattr(chunk, "content", ""))
            if text:
                if time_to_first_token is None:
                    time_to_first_token = time.perf_counter() - started
                    log_progress(
                        f"First token received after {time_to_first_token:.2f}s."
                    )
                response_chunks.append(text)
        latency = time.perf_counter() - started
        answer = "".join(response_chunks)
        if not answer and merged_chunk is not None:
            answer = _llm_content_to_text(getattr(merged_chunk, "content", ""))
        output = answer.strip()
        output_char_count = len(output)
        estimated_output_tokens = estimate_token_count(output)
        return {
            "answer": output,
            "generation_latency_seconds": latency,
            "time_to_first_token_seconds": time_to_first_token,
            "output_char_count": output_char_count,
            "estimated_output_tokens": estimated_output_tokens,
            "output_chars_per_second": safe_rate(
                numerator=output_char_count,
                denominator_seconds=latency,
            ),
            "estimated_output_tokens_per_second": safe_rate(
                numerator=estimated_output_tokens,
                denominator_seconds=latency,
            ),
            "generation_error": None,
        }
    except Exception as exc:
        latency = time.perf_counter() - started
        return {
            "answer": "",
            "generation_latency_seconds": latency,
            "time_to_first_token_seconds": time_to_first_token,
            "output_char_count": 0,
            "estimated_output_tokens": 0,
            "output_chars_per_second": 0.0,
            "estimated_output_tokens_per_second": 0.0,
            "generation_error": f"{type(exc).__name__}: {exc}",
        }


def _run_warmups(
    *,
    model: Any,
    warmup_generations: int,
) -> None:
    for index in range(warmup_generations):
        print(f"Warmup generation {index + 1}/{warmup_generations}...")
        _stream_generation(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            generation_prompt="Question:\nWarm up.\n\nContext:\nWarm up.\n\nProvide a concise answer grounded in the context.",
        )


def run_generation_latency_benchmark(
    *,
    chunk_sizes: tuple[int, ...] = DEFAULT_CHUNK_SIZES,
    chunk_overlap_ratio: float = 0.2,
    chunk_overlap: int | None = None,
    retrieval_mode: str = "similarity",
    similarity_k: int = 8,
    bm25_k: int = 35,
    final_k: int = 10,
    rrf_k: int = 80,
    limit_contracts: int = 1000,
    limit_questions_per_contract: int = 1000,
    dataset_path: Path | None = None,
    corpus_dir: Path | None = None,
    chunking_strategy: ChunkingStrategy = DEFAULT_CHUNKING_STRATEGY,
    chat_model_id: str = DEFAULT_CHAT_MODEL_ID,
    embedding_model_id: str = DEFAULT_EMBEDDING_MODEL_ID,
    warmup_generations: int = 1,
    use_reranker: bool = False,
) -> None:
    os.environ.setdefault("LANGCHAIN_TRACING", "false")

    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR
    vector_cache_dir = (
        DEFAULT_BASELINE_VECTOR_CACHE_DIR
        if chunking_strategy == "langchain_basic"
        else DEFAULT_VECTOR_CACHE_DIR
    )

    started_at = datetime.now(timezone.utc)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = DEFAULT_OUTPUT_DIR / f"generation_latency_results_{timestamp}.json"

    contexts = _load_grouped_contexts(
        dataset_path=dataset_path,
        limit_contracts=limit_contracts,
    )
    total_questions = sum(
        len(context_tests[:limit_questions_per_contract])
        for _, context_tests in contexts
    )

    try:
        if _requires_vector_retrieval(retrieval_mode):
            if not pipeline.check_embedding_model_connection(embedding_model_id):
                log_progress(
                    "Embedding model connection check failed; exiting benchmark."
                )
                print(
                    f"Embedding model connection failed for {embedding_model_id}. "
                    "Please start the local embedding server."
                )
                return
        if not check_chat_model_connection(chat_model_id):
            log_progress("Chat model connection check failed; exiting benchmark.")
            print(
                f"Chat model connection failed for {chat_model_id}. "
                "Please start the local chat model server."
            )
            return
        chat_model = get_chat_model(chat_model_id)
        embedding_model = (
            pipeline.get_embeddings_model(embedding_model_id)
            if _requires_vector_retrieval(retrieval_mode)
            else None
        )
    except Exception as exc:
        log_progress("Connection setup failed; exiting benchmark.")
        print(f"Connection check failed: {exc}")
        print("Please check your local model configuration. Benchmark will exit now.")
        return

    log_progress(
        f"Dataset: {dataset_path} ({len(contexts)} contexts, {total_questions} questions)"
    )
    log_progress(f"Corpus dir: {corpus_dir}")
    log_progress(f"Output path: {output_path}")
    log_progress(f"Chat model: {chat_model_id}")
    log_progress(f"Embedding model: {embedding_model_id}")
    log_progress(
        "Retrieval mode: "
        f"{retrieval_mode} (similarity_k={similarity_k}, bm25_k={bm25_k}, "
        f"final_k={final_k}, rrf_k={rrf_k}, use_reranker={use_reranker})"
    )
    log_progress(f"Chunk sizes: {chunk_sizes}")
    log_progress(
        "Chunk overlap: "
        f"{chunk_overlap if chunk_overlap is not None else f'{chunk_overlap_ratio:.2f} ratio'}"
    )

    if warmup_generations > 0:
        _run_warmups(model=chat_model, warmup_generations=warmup_generations)

    if sys.stdin.isatty():
        log_progress("Press Enter to start the benchmark...")
        input()
    else:
        log_progress("Non-interactive mode detected; starting benchmark immediately.")

    all_results: list[dict[str, Any]] = []
    summaries_by_chunk_size: dict[str, dict[str, Any]] = {}
    interrupted = False
    interruption_message: str | None = None

    try:
        for chunk_size in chunk_sizes:
            overlap = resolve_chunk_overlap(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunk_overlap_ratio=chunk_overlap_ratio,
            )
            chunking_config = chunker.ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
            chunk_results: list[dict[str, Any]] = []

            log_progress(
                f"\nRunning chunk_size={chunk_size}, chunk_overlap={overlap}..."
            )

            for context_index, (file_path, context_tests) in enumerate(contexts):
                context_tests = context_tests[:limit_questions_per_contract]
                corpus_file = corpus_dir / file_path
                corpus_text = corpus_file.read_text(encoding="utf-8")

                log_progress(
                    f"Preparing chunks/vector store for context {context_index + 1}/{len(contexts)}: {file_path}"
                )
                embedded_database = _build_embedded_database(
                    file_path=file_path,
                    corpus_text=corpus_text,
                    retrieval_mode=retrieval_mode,
                    embedding_model=embedding_model,
                    embedding_model_id=embedding_model_id,
                    vector_cache_dir=vector_cache_dir,
                    chunking_config=chunking_config,
                    chunking_strategy=chunking_strategy,
                )

                total_in_context = len(context_tests)
                for question_index, test in enumerate(context_tests, start=1):
                    query = test["query"]
                    question_id = hashlib.md5(query.encode()).hexdigest()[:12]
                    log_progress(
                        f"Chunk {chunk_size} | Context {context_index + 1}/{len(contexts)} "
                        f"Q {question_index}/{total_in_context}: {query[:100]}"
                    )

                    log_progress("Running retrieval...")
                    retrieval_result = _run_retrieval_only(
                        query=query,
                        embedded_database=embedded_database,
                        chat_model_id=chat_model_id,
                        retrieval_mode=retrieval_mode,
                        similarity_k=similarity_k,
                        bm25_k=bm25_k,
                        final_k=final_k,
                        rrf_k=rrf_k,
                        use_reranker=use_reranker,
                    )
                    retrieved_docs = retrieval_result.get("retrieved_docs") or []
                    generation_prompt, context_text = _build_generation_prompt(
                        query=query,
                        retrieved_docs=retrieved_docs,
                    )
                    log_progress(
                        "Running generation... "
                        f"(retrieved_chunks={len(retrieved_docs)}, "
                        f"prompt_chars={len(generation_prompt)}, "
                        f"context_chars={len(context_text)})"
                    )
                    generation_result = _stream_generation(
                        model=chat_model,
                        system_prompt=SYSTEM_PROMPT,
                        generation_prompt=generation_prompt,
                    )
                    prompt_char_count = len(generation_prompt)
                    context_char_count = len(context_text)

                    row = {
                        "chunk_size": chunk_size,
                        "chunk_overlap": overlap,
                        "file_path": file_path,
                        "question_id": question_id,
                        "query": query,
                        "generation_latency_seconds": generation_result[
                            "generation_latency_seconds"
                        ],
                        "time_to_first_token_seconds": generation_result[
                            "time_to_first_token_seconds"
                        ],
                        "prompt_char_count": prompt_char_count,
                        "context_char_count": context_char_count,
                        "output_char_count": generation_result["output_char_count"],
                        "output_chars_per_second": generation_result[
                            "output_chars_per_second"
                        ],
                        "estimated_prompt_tokens": estimate_token_count(
                            generation_prompt
                        ),
                        "estimated_output_tokens": generation_result[
                            "estimated_output_tokens"
                        ],
                        "estimated_output_tokens_per_second": generation_result[
                            "estimated_output_tokens_per_second"
                        ],
                        "retrieved_chunk_count": len(retrieved_docs),
                        "predicted_answer": generation_result["answer"],
                        "generation_error": generation_result["generation_error"],
                    }
                    chunk_results.append(row)
                    all_results.append(row)
                    log_progress(
                        "Completed question. "
                        f"latency={row['generation_latency_seconds']:.2f}s, "
                        f"ttft={row['time_to_first_token_seconds'] if row['time_to_first_token_seconds'] is not None else 'n/a'}"
                    )

            summaries_by_chunk_size[str(chunk_size)] = summarize_chunk_results(
                chunk_results
            )
    except KeyboardInterrupt as exc:
        interrupted = True
        interruption_message = str(exc) or "KeyboardInterrupt"
        print("\nBenchmark interrupted. Saving partial results...")

    summary = {
        "run": {
            "status": "interrupted" if interrupted else "completed",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "output_path": str(output_path),
        },
        "dataset": {
            "dataset_path": str(dataset_path),
            "corpus_dir": str(corpus_dir),
            "contexts_evaluated": len(contexts),
            "questions_per_chunk_size": total_questions,
        },
        "config": {
            "chat_model_id": chat_model_id,
            "embedding_model_id": embedding_model_id,
            "retrieval_mode": retrieval_mode,
            "use_reranker": use_reranker,
            "retrieval_params": {
                "similarity_k": similarity_k,
                "bm25_k": bm25_k,
                "final_k": final_k,
                "rrf_k": rrf_k,
            },
            "chunking": {
                "strategy": chunking_strategy,
                "chunk_sizes": list(chunk_sizes),
                "chunk_overlap": chunk_overlap,
                "chunk_overlap_ratio": chunk_overlap_ratio,
            },
            "warmup_generations": warmup_generations,
            "storage": {
                "output_dir": str(DEFAULT_OUTPUT_DIR),
                "vector_cache_dir": str(vector_cache_dir),
            },
        },
        "metrics_by_chunk_size": summaries_by_chunk_size,
    }
    if interruption_message is not None:
        summary["run"]["interruption_message"] = interruption_message

    atomic_write_json(output_path, {"summary": summary, "results": all_results})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure local-model generation latency across chunk sizes."
    )
    parser.add_argument(
        "--chunk-sizes",
        type=parse_chunk_sizes,
        default=DEFAULT_CHUNK_SIZES,
        help="Comma-separated chunk sizes to benchmark.",
    )
    parser.add_argument(
        "--chunk-overlap-ratio",
        type=float,
        default=0.2,
        help="Chunk overlap ratio used when --chunk-overlap is not provided.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Fixed chunk overlap. Overrides --chunk-overlap-ratio.",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="similarity",
        choices=["similarity", "bm25", "hybrid"],
        help="Retrieval mode used to build realistic generation context.",
    )
    parser.add_argument(
        "--similarity-k",
        type=int,
        default=8,
        help="Top-k for dense retrieval.",
    )
    parser.add_argument(
        "--bm25-k",
        type=int,
        default=35,
        help="Top-k for BM25 retrieval.",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=10,
        help="Final number of retrieved chunks included in generation context.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=80,
        help="RRF smoothing constant for hybrid retrieval.",
    )
    parser.add_argument(
        "--limit-contracts",
        type=int,
        default=1000,
        help="Maximum number of contexts to evaluate.",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=1000,
        help="Maximum number of questions per context.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to a LegalBench-RAG sample JSON.",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Root directory containing corpus text files referenced by snippets.",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=["legal", "langchain_basic"],
        help="Chunking strategy to benchmark.",
    )
    parser.add_argument(
        "--chat-model-id",
        type=str,
        default=DEFAULT_CHAT_MODEL_ID,
        help="Chat model id from the model registry.",
    )
    parser.add_argument(
        "--embedding-model-id",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL_ID,
        help="Embedding model id from the embedding registry.",
    )
    parser.add_argument(
        "--warmup-generations",
        type=int,
        default=1,
        help="Untimed generation calls to run before collecting measurements.",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup_generations",
        action="store_const",
        const=0,
        help="Disable untimed warmup generations.",
    )
    parser.add_argument(
        "--use-reranker",
        dest="use_reranker",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranker while building context.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_generation_latency_benchmark(
        chunk_sizes=args.chunk_sizes,
        chunk_overlap_ratio=args.chunk_overlap_ratio,
        chunk_overlap=args.chunk_overlap,
        retrieval_mode=args.retrieval_mode,
        similarity_k=args.similarity_k,
        bm25_k=args.bm25_k,
        final_k=args.final_k,
        rrf_k=args.rrf_k,
        limit_contracts=args.limit_contracts,
        limit_questions_per_contract=args.limit_questions,
        dataset_path=args.dataset,
        corpus_dir=args.corpus_dir,
        chunking_strategy=args.chunking_strategy,
        chat_model_id=args.chat_model_id,
        embedding_model_id=args.embedding_model_id,
        warmup_generations=args.warmup_generations,
        use_reranker=args.use_reranker,
    )


if __name__ == "__main__":
    main()
