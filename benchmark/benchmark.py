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

from powerllm.models.model_resolver import check_chat_model_connection
import powerllm.retrieval.chunker as chunker
import powerllm.retrieval.pipeline as pipeline
from benchmark.benchmark_pipeline import (
    get_or_create_persistent_vector_store,
    get_answer,
)
from benchmark.chunking_strategies import (
    DEFAULT_CHUNKING_STRATEGY,
    ChunkingStrategy,
    build_benchmark_chunks,
)
from benchmark.benchmark_metrics import (
    DEFAULT_HIT_KS,
    DEFAULT_PRECISION_KS,
    DEFAULT_RECALL_KS,
    aggregate_retrieval_results,
    evaluate_retrieval_question,
)
from benchmark.benchmark_utils import (
    annotate_documents,
    atomic_write_json,
    calculate_latency,
    precompute_bm25_queries_if_needed,
    short_string,
)


DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "legalbench_sample_cuad_q200_c20_min10_seed42.json"
DEFAULT_CORPUS_DIR = Path(__file__).resolve().parent / "corpus"
DEFAULT_OUTPUT_DIR = Path("benchmark_results")
DEFAULT_VECTOR_CACHE_DIR = Path("benchmark/chroma_cache")
DEFAULT_BASELINE_VECTOR_CACHE_DIR = Path("benchmark/chroma_cache_langchain_basic")
DEFAULT_BM25_QUERY_CACHE_DIR = Path("benchmark/query_cache")

BENCHMARK_SYSTEM_PROMPT = (
    "Answer using only the exact text span(s) from the context."
    "Do not explain."
    "Do not cite sections."
    "If multiple spans are relevant, return them as a semicolon-separated list."
    "If no answer is found, return: NOT FOUND"
)


def parse_k_values(raw_value: str) -> tuple[int, ...]:
    values = []
    for part in raw_value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        try:
            value = int(stripped)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid k value '{stripped}'. Expected comma-separated integers."
            ) from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(
                f"Invalid k value '{value}'. K values must be positive integers."
            )
        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(
            "Expected at least one positive integer k value."
        )

    return tuple(dict.fromkeys(values))


def set_benchmark_goal(
    goal: str,
    llm_mode: str,
    *,
    retrieval_mode: str,
    chunking_strategy: ChunkingStrategy = DEFAULT_CHUNKING_STRATEGY,
    chunking_config: chunker.ChunkingConfig | None = None,
    similarity_k: int,
    bm25_k: int,
    final_k: int,
    rrf_k: int,
    recall_ks: tuple[int, ...],
    hit_ks: tuple[int, ...],
    precision_ks: tuple[int, ...],
    limit_contracts: int,
    limit_questions_per_contract: int,
    dataset_path: Path | None = None,
    corpus_dir: Path | None = None,
    short_retrieved_chunk_content: bool = True,
    use_reranker: bool = False,
):
    """Route to the appropriate benchmark pipeline.

    :param goal: Benchmark goal — only "retrieval" is currently supported.
    :param llm_mode: BM25 query rewrite mode: "standard", "batch", or "concurrent".
    :param dataset_path: Path to a LegalBench-RAG sample JSON produced by dataset_cut3.py.
    :param corpus_dir: Root directory containing the corpus text files referenced by file_path.
    """
    goal = "retrieval"  # only retrieval is supported for now
    if llm_mode not in {"standard", "batch", "concurrent"}:
        raise ValueError("llm_mode must be one of: 'standard', 'batch', 'concurrent'.")

    match goal:
        case "retrieval":
            print(f"Running retrieval benchmark... (llm_mode={llm_mode})")
            run_retrieval_benchmark(
                llm_mode=llm_mode,
                retrieval_mode=retrieval_mode,
                chunking_strategy=chunking_strategy,
                chunking_config=chunking_config,
                similarity_k=similarity_k,
                bm25_k=bm25_k,
                final_k=final_k,
                rrf_k=rrf_k,
                recall_ks=recall_ks,
                hit_ks=hit_ks,
                precision_ks=precision_ks,
                limit_contracts=limit_contracts,
                limit_questions_per_contract=limit_questions_per_contract,
                dataset_path=dataset_path,
                corpus_dir=corpus_dir,
                short_retrieved_chunk_content=short_retrieved_chunk_content,
                use_reranker=use_reranker,
            )
        case "generation":
            print("Running generation benchmark... (not supported yet)")
        case "end-to-end":
            print("Running end-to-end benchmark... (not supported yet)")
        case _:
            raise ValueError(f"Unsupported benchmark goal: {goal}")

def run_retrieval_benchmark(
    *,
    llm_mode: str = "standard",
    retrieval_mode: str = "hybrid",
    chunking_strategy: ChunkingStrategy = DEFAULT_CHUNKING_STRATEGY,
    chunking_config: chunker.ChunkingConfig | None = None,
    similarity_k: int,
    bm25_k: int,
    final_k: int,
    rrf_k: int,
    recall_ks: tuple[int, ...],
    hit_ks: tuple[int, ...],
    precision_ks: tuple[int, ...],
    limit_contracts: int,
    limit_questions_per_contract: int = 1000,
    dataset_path: Path | None = None,
    corpus_dir: Path | None = None,
    short_retrieved_chunk_content: bool = False,
    use_reranker: bool = False,
) -> None:
    os.environ.setdefault("LANGCHAIN_TRACING", "false")

    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    corpus_dir = corpus_dir or DEFAULT_CORPUS_DIR
    output_dir = DEFAULT_OUTPUT_DIR
    vector_cache_dir = (
        DEFAULT_BASELINE_VECTOR_CACHE_DIR
        if chunking_strategy == "langchain_basic"
        else DEFAULT_VECTOR_CACHE_DIR
    )
    bm25_query_cache_dir = DEFAULT_BM25_QUERY_CACHE_DIR
    chunking_config = chunking_config or chunker.ChunkingConfig()

    embedding_model_id = "Qwen3-Embedding-0.6B-4bit-DWQ-Online" # "Qwen3-Embedding-0.6B-4bit-DWQ" for local testing
    chat_model_id = "qwen-3.5-9b-openrouter" # "gemini-3.1-flash-lite" / "qwen-3.5-9b-openrouter"

    started_at = datetime.now(timezone.utc)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"benchmark_results_{timestamp}.json"

    # Load LegalBench-RAG dataset
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    tests: list[dict[str, Any]] = data["tests"]

    # Group tests by their primary context file_path
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for test in tests:
        fp = test["snippets"][0]["file_path"]
        grouped[fp].append(test)

    contexts = list(grouped.items())
    if limit_contracts < len(contexts):
        contexts = contexts[:limit_contracts]

    bm25_query_cache_path = bm25_query_cache_dir / "bm25_query_cache_all.json"

    all_queries = [
        test["query"]
        for _, context_tests in contexts
        for test in context_tests
    ]
    try:
        bm25_query_cache = precompute_bm25_queries_if_needed(
            queries=all_queries,
            llm_mode=llm_mode,
            chat_model_id=chat_model_id,
            cache_path=bm25_query_cache_path,
        )
    except KeyboardInterrupt:
        print("\nBenchmark interrupted during BM25 rewrite precompute.")
        return

    try:
        pipeline.check_embedding_model_connection(embedding_model_id)
        check_chat_model_connection(chat_model_id)
    except Exception as e:
        print(f"Connection check failed: {e}")
        print("Please check your model configuration. Benchmark will exit now.")
        return

    use_vector_retrieval = retrieval_mode in {"similarity", "hybrid"}
    embedding_model = (
        pipeline.get_embeddings_model(embedding_model_id)
        if use_vector_retrieval
        else None
    )

    total_contexts = len(contexts)
    print(f"Dataset: {dataset_path} ({total_contexts} contexts, {len(all_queries)} questions)")
    print(f"Corpus dir: {corpus_dir}")
    print(f"Output path: {output_path}")
    print(f"Embedding model: {embedding_model_id}")
    print(f"Chat model: {chat_model_id}")
    print(
        "Retrieval mode: "
        f"{retrieval_mode} (similarity_k={similarity_k}, bm25_k={bm25_k}, "
        f"final_k={final_k}, rrf_k={rrf_k}, use_reranker={use_reranker})"
    )
    print(f"Metric k config: recall_ks={recall_ks}, hit_ks={hit_ks}, precision_ks={precision_ks}")
    print(
        "Chunking config: "
        f"(strategy={chunking_strategy}, chunk_size={chunking_config.chunk_size}, "
        f"chunk_overlap={chunking_config.chunk_overlap})"
    )
    print(f"LLM mode: {llm_mode}")
    print(f"BM25 query cache: {bm25_query_cache_path} ({len(bm25_query_cache)} cached)")

    if sys.stdin.isatty():
        print("Press Enter to start the benchmark...")
        input()
    else:
        print("Non-interactive mode detected; starting benchmark immediately.")

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    interrupted = False
    interruption_message: str | None = None

    try:
        for context_index, (file_path, context_tests) in enumerate(contexts):
            context_tests = context_tests[:limit_questions_per_contract]

            corpus_file = corpus_dir / file_path
            corpus_text = corpus_file.read_text(encoding="utf-8")

            if use_vector_retrieval:
                if embedding_model is None:
                    raise ValueError(
                        "An embedding model is required for similarity or hybrid retrieval."
                    )
                embedded_database: Any = get_or_create_persistent_vector_store(
                    title=file_path,
                    context=corpus_text,
                    embedding_model=embedding_model,
                    embedding_model_id=embedding_model_id,
                    cache_root=vector_cache_dir,
                    chunking_config=chunking_config,
                    chunking_strategy=chunking_strategy,
                )
            else:
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
                embedded_database = annotate_documents(context=corpus_text, documents=docs)

            total_in_context = len(context_tests)
            for question_index, test in enumerate(context_tests, start=1):
                query = test["query"]
                question_id = hashlib.md5(query.encode()).hexdigest()[:12]

                # Answer annotations from snippets belonging to this context file
                answer_annotations = [
                    {"answer_start": s["span"][0], "answer_end": s["span"][1]}
                    for s in test["snippets"]
                    if s.get("file_path") == file_path
                ]

                print(
                    f"Context {context_index + 1}/{total_contexts} "
                    f"Q {question_index}/{total_in_context}: {query[:100]}"
                )

                started = time.perf_counter()

                cached_bm25 = bm25_query_cache.get(query, {})
                bm25_query_override = cached_bm25.get("bm25_query")
                bm25_query_source = (
                    "cache"
                    if isinstance(bm25_query_override, str) and bm25_query_override.strip()
                    else "live"
                )

                # For now we skip generation and test retrieval only
                answer_result = get_answer(
                    query=query,
                    embedded_database=embedded_database,
                    chat_model_id=chat_model_id,
                    system_prompt=BENCHMARK_SYSTEM_PROMPT,
                    retrieval_mode=retrieval_mode,
                    similarity_k=similarity_k,
                    bm25_k=bm25_k,
                    final_k=final_k,
                    rrf_k=rrf_k,
                    bm25_query_override=(
                        bm25_query_override
                        if isinstance(bm25_query_override, str)
                        else None
                    ),
                    run_mode="retrieval_only",
                    use_reranker=use_reranker,
                )

                retrieved_chunks = answer_result["retrieved_chunks"]
                if short_retrieved_chunk_content:
                    for inner_chunk in retrieved_chunks:
                        inner_chunk["content"] = short_string(
                            inner_chunk["content"],
                            15,
                        )
                        # Other fields are kept the same

                bm25_query = answer_result.get("bm25_query")
                bm25_query_rewrite_applied = (
                    cached_bm25.get("bm25_query_rewrite_applied")
                    if bm25_query_source == "cache"
                    else answer_result.get("bm25_query_rewrite_applied", False)
                )
                bm25_query_rewrite_error = (
                    cached_bm25.get("bm25_query_rewrite_error")
                    if bm25_query_source == "cache"
                    else answer_result.get("bm25_query_rewrite_error")
                )

                latency_seconds = calculate_latency(started)
                latencies.append(latency_seconds)

                metrics = evaluate_retrieval_question(
                    answer_annotations,
                    retrieved_chunks,
                    final_k=final_k,
                    recall_ks=recall_ks,
                    hit_ks=hit_ks,
                    precision_ks=precision_ks,
                )

                results.append(
                    {
                        "file_path": file_path,
                        "question_id": question_id,
                        "query": query,
                        "bm25_query": bm25_query,
                        "bm25_query_source": bm25_query_source,
                        "bm25_query_rewrite_applied": bm25_query_rewrite_applied,
                        "bm25_query_rewrite_error": bm25_query_rewrite_error,
                        "answer_annotations": answer_annotations,
                        "retrieved_chunks": retrieved_chunks,
                        "final_retrieved_chunks": retrieved_chunks,
                        "coverage": metrics["coverage"],
                        "has_valid_gold_spans": metrics["has_valid_gold_spans"],
                        "retrieval_metrics": metrics["retrieval_metrics"],
                        "failure_analysis": metrics["failure_analysis"],
                        "latency_seconds": latency_seconds,
                    }
                )
    except KeyboardInterrupt as exc:
        interrupted = True
        interruption_message = str(exc) or "KeyboardInterrupt"
        print("\nBenchmark interrupted. Saving partial results...")

    aggregated_results = aggregate_retrieval_results(
        results,
        recall_ks=recall_ks,
        hit_ks=hit_ks,
        precision_ks=precision_ks,
    )

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
            "contexts_evaluated": total_contexts,
            "questions_evaluated": aggregated_results["total_questions"],
            "answerable_questions": aggregated_results["answerable_questions"],
            "no_answer_questions": aggregated_results["no_answer_questions"],
        },
        "config": {
            "embedding_model_id": embedding_model_id,
            "chat_model_id": chat_model_id,
            "retrieval_mode": retrieval_mode,
            "use_reranker": use_reranker,
            "retrieval_params": {
                "similarity_k": similarity_k,
                "bm25_k": bm25_k,
                "final_k": final_k,
                "rrf_k": rrf_k,
            },
            "metric_k_config": {
                "recall_ks": list(recall_ks),
                "hit_ks": list(hit_ks),
                "precision_ks": list(precision_ks),
            },
            "chunking": {
                "strategy": chunking_strategy,
                "chunk_size": chunking_config.chunk_size,
                "chunk_overlap": chunking_config.chunk_overlap,
            },
            "query_rewrite": {
                "llm_mode": llm_mode,
                "bm25_query_cache_path": str(bm25_query_cache_path),
            },
            "storage": {
                "output_dir": str(output_dir),
                "vector_cache_dir": str(vector_cache_dir),
            },
        },
        "metrics": {
            **aggregated_results["metrics"],
            "latency": {
                "avg_seconds": statistics.mean(latencies) if latencies else 0.0,
            },
        },
        "coverage": aggregated_results["coverage"],
        "failure_analysis": aggregated_results["failure_analysis"],
    }
    if interruption_message is not None:
        summary["run"]["interruption_message"] = interruption_message

    atomic_write_json(output_path, {"summary": summary, "results": results})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved results to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval benchmark.")
    parser.add_argument(
        "--llm-mode",
        type=str,
        default="concurrent",
        choices=["batch", "standard", "concurrent"],
        help="BM25 query rewrite mode.",
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default=DEFAULT_CHUNKING_STRATEGY,
        choices=["legal", "langchain_basic"],
        help="Chunking strategy to benchmark.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size for document splitting.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=500,
        help="Chunk overlap for document splitting.",
    )
    parser.add_argument(
        "--retrieval-mode",
        type=str,
        default="hybrid",
        choices=["similarity", "bm25", "hybrid"],
        help="Retrieval mode to benchmark.",
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
        help="Final top-k after retrieval fusion.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=80,
        help="RRF smoothing constant for hybrid retrieval.",
    )
    parser.add_argument(
        "--recall-ks",
        type=parse_k_values,
        default=DEFAULT_RECALL_KS,
        help="Comma-separated k values for recall metrics.",
    )
    parser.add_argument(
        "--hit-ks",
        type=parse_k_values,
        default=DEFAULT_HIT_KS,
        help="Comma-separated k values for hit metrics.",
    )
    parser.add_argument(
        "--precision-ks",
        type=parse_k_values,
        default=DEFAULT_PRECISION_KS,
        help="Comma-separated k values for precision metrics.",
    )
    parser.add_argument(
        "--limit-contracts",
        type=int,
        default=1000,
        help="Maximum number of contexts (documents) to evaluate.",
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
        help="Path to a LegalBench-RAG sample JSON (produced by dataset_cut3.py).",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=DEFAULT_CORPUS_DIR,
        help="Root directory containing corpus text files referenced by file_path in snippets.",
    )
    parser.add_argument(
        "--no-short-retrieved-chunk-content",
        dest="short_retrieved_chunk_content",
        action="store_false",
        default=True,
        help="""
        Disable shortening retrieved chunk content. By default, content is shortened. e.g:
             "retrieved_chunks": [
                {
                    "content": "Any information included in your publicly visible 
                    ... 
                    "title": "privacy_qa/Keep.txt"
                },
            ]
            ->
            "retrieved_chunks": [
                {
                    "content": "Any information included in your publicly visible ...",
                    ... # keep the same fields
                    "title": "privacy_qa/Keep.txt"
                },
            ]
        """,
    )

    parser.add_argument(
        "--use-reranker",
        dest="use_reranker",
        action="store_true",
        default=False,
        help="Enable cross-encoder reranker (default: disabled).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    chunking_config = chunker.ChunkingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    set_benchmark_goal(
        goal="retrieval",
        llm_mode=args.llm_mode,
        retrieval_mode=args.retrieval_mode,
        chunking_strategy=args.chunking_strategy,
        chunking_config=chunking_config,
        similarity_k=args.similarity_k,
        bm25_k=args.bm25_k,
        final_k=args.final_k,
        rrf_k=args.rrf_k,
        recall_ks=args.recall_ks,
        hit_ks=args.hit_ks,
        precision_ks=args.precision_ks,
        limit_contracts=args.limit_contracts,
        limit_questions_per_contract=args.limit_questions,
        dataset_path=args.dataset,
        corpus_dir=args.corpus_dir,
        short_retrieved_chunk_content=args.short_retrieved_chunk_content,
        use_reranker=args.use_reranker,
    )


if __name__ == "__main__":
    main()
