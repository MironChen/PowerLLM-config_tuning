"""Shared utilities for the retrieval benchmark."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

import powerllm.retrieval.chunker as chunker

# ---------------------------------------------------------------------------
# Document chunking utilities
# ---------------------------------------------------------------------------
def annotate_documents(
    *,
    context: str,
    documents: list[Document],
    title: str | None = None,
) -> list[Document]:
    """Add start/end character offsets using each chunk's original source text.

    The title parameter is accepted for API compatibility with CUAD-format
    callers that pass it from the grouped tuning dataset shape, but offsets
    are derived solely from the shared source context.
    """
    del title
    search_from = 0
    for doc in documents:
        text = chunker.get_chunk_source_text(doc)

        idx = context.find(text, search_from)
        if idx < 0:
            raise ValueError("Benchmark chunk source text could not be aligned to context")
        doc.metadata["start"] = idx
        doc.metadata["end"] = idx + len(text)
        search_from = idx
    return documents


def build_benchmark_collection_name(
    *,
    title: str,
    context: str,
    chunking_strategy: str = "legal",
) -> str:
    digest = hashlib.sha256(context.encode()).hexdigest()[:12]
    strategy_digest = hashlib.sha256(chunking_strategy.encode()).hexdigest()[:6]
    return f"bench_{strategy_digest}_{digest}"

# ---------------------------------------------------------------------------
# Chroma vector store caching
# ---------------------------------------------------------------------------


def build_benchmark_cache_dir(
    *,
    title: str,
    context: str,
    embedding_model_id: str,
    cache_root: Path,
    chunking_config: Any,
    chunking_strategy: str = "legal",
) -> Path:
    content_hash = hashlib.sha256(context.encode()).hexdigest()[:16]
    model_hash = hashlib.sha256(embedding_model_id.encode()).hexdigest()[:8]
    # Include a schema/version tag so benchmark cache rebuilds when chunk
    # metadata required by evaluation changes (for example start/end offsets).
    cfg = (
        f"{chunking_strategy}_"
        f"{chunking_config.chunk_size}_{chunking_config.chunk_overlap}_offsets_v2"
    )
    cfg_hash = hashlib.sha256(cfg.encode()).hexdigest()[:8]
    return Path(cache_root) / f"{content_hash}_{model_hash}_{cfg_hash}"


# ---------------------------------------------------------------------------
# Retrieved document serialization
# ---------------------------------------------------------------------------


def serialize_retrieved_docs(docs: list[Document]) -> list[dict[str, Any]]:
    result = []
    for i, doc in enumerate(docs):
        entry: dict[str, Any] = {
            "content": doc.page_content,
            "start": doc.metadata.get("start"),
            "end": doc.metadata.get("end"),
            "rank": doc.metadata.get("rank", i + 1),
        }
        for k, v in doc.metadata.items():
            if k not in entry:
                entry[k] = v
        result.append(entry)
    return result


# ---------------------------------------------------------------------------
# Span matching (used by benchmark_metrics)
# ---------------------------------------------------------------------------


def _chunk_covers_span(
    chunk: dict[str, Any],
    *,
    answer_start: int,
    answer_end: int,
) -> bool:
    start = chunk.get("start")
    end = chunk.get("end")
    if not isinstance(start, int) or not isinstance(end, int):
        return False
    return start <= answer_start and end >= answer_end


def filter_valid_expected_answer_spans(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        a
        for a in annotations
        if (
            isinstance(a.get("answer_start"), int)
            and isinstance(a.get("answer_end"), int)
            and a["answer_start"] >= 0
            and a["answer_end"] > a["answer_start"]
        )
    ]


def build_cuad_answer_annotations(
    answers: list[dict[str, Any]],
) -> list[dict[str, int]]:
    """Convert CUAD-style answers into answer span annotations."""
    annotations: list[dict[str, int]] = []
    for answer in answers:
        answer_start = answer.get("answer_start")
        text = answer.get("text")
        if not isinstance(answer_start, int) or answer_start < 0:
            continue
        if not isinstance(text, str) or not text:
            continue
        annotations.append(
            {
                "answer_start": answer_start,
                "answer_end": answer_start + len(text),
            }
        )
    return annotations


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def calculate_latency(started: float) -> float:
    return time.perf_counter() - started


# Cache key delimiter - changing this invalidates all existing cache entries
BM25_CACHE_KEY_DELIMITER = "::"


def build_bm25_cache_key(
    *,
    title: str,
    question_id: str,
    question: str,
) -> str:
    return f"{title}{BM25_CACHE_KEY_DELIMITER}{question_id}{BM25_CACHE_KEY_DELIMITER}{question}"


def load_benchmark_records(
    dataset_path: Path,
    *,
    limit_contracts: int = -1,
    limit_questions_per_contract: int = -1,
) -> list[dict[str, Any]]:
    """Load benchmark data into the grouped {title, context, qas} tuning shape."""
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_records = json.load(f)

    grouped_records: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, record in enumerate(raw_records):
        title = record.get("title")
        context = record.get("context")
        question = record.get("question")
        answers = record.get("answers")
        if not isinstance(title, str) or not isinstance(context, str):
            continue
        if not isinstance(question, str) or not isinstance(answers, list):
            continue

        key = (title, context)
        grouped_records.setdefault(key, []).append(
            {
                "id": str(record.get("id") or f"{title}_{index}"),
                "question": question,
                "answers": answers,
                "is_impossible": bool(record.get("is_impossible", False)),
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


# ---------------------------------------------------------------------------
# BM25 query rewrite cache
# ---------------------------------------------------------------------------


def precompute_bm25_queries_if_needed(
    *,
    queries: list[str] | None = None,
    dataset: list[dict[str, Any]] | None = None,
    llm_mode: str,
    chat_model_id: str,
    cache_path: Path,
) -> dict[str, dict]:
    """Pre-compute BM25-optimized query rewrites and persist them to a cache file.

    Returns the full cache dict mapping query → {bm25_query, bm25_query_rewrite_applied,
    bm25_query_rewrite_error}. Queries already in the cache are skipped.
    """
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: Could not load cache from {cache_path}: {exc}")
            cache = {}

    cache_inputs: list[tuple[str, str]] = []
    if dataset is not None:
        for contract in dataset:
            title = contract.get("title")
            if not isinstance(title, str):
                continue
            for qa in contract.get("qas", []):
                question = qa.get("question")
                if not isinstance(question, str):
                    continue
                cache_key = build_bm25_cache_key(
                    title=title,
                    question_id=str(qa.get("id", "")),
                    question=question,
                )
                cache_inputs.append((cache_key, question))
    elif queries is not None:
        cache_inputs = [(query, query) for query in queries]
    else:
        raise ValueError("Either queries or dataset must be provided.")

    unique_inputs = list(dict.fromkeys(cache_inputs))
    uncached = [
        (cache_key, query)
        for cache_key, query in unique_inputs
        if cache_key not in cache
    ]
    if not uncached:
        return cache

    print(f"Pre-computing BM25 rewrites for {len(uncached)} uncached queries...")

    from powerllm.models.model_resolver import get_chat_model
    from powerllm.retrieval.query_builder import QueryBuilder

    chat_model = get_chat_model(chat_model_id)
    qb = QueryBuilder(vector_store=None, chat_model=chat_model)

    def _rewrite(query: str) -> dict:
        try:
            bm25_query = qb.gen_keywords_str(query)
            return {
                "bm25_query": bm25_query,
                "bm25_query_rewrite_applied": True,
                "bm25_query_rewrite_error": None,
            }
        except Exception as exc:
            return {
                "bm25_query": None,
                "bm25_query_rewrite_applied": False,
                "bm25_query_rewrite_error": str(exc),
            }

    interrupted = False
    if llm_mode == "concurrent":
        pool = ThreadPoolExecutor(max_workers=10)
        try:
            future_to_query = {
                pool.submit(_rewrite, query): cache_key
                for cache_key, query in uncached
            }
            for future in as_completed(future_to_query):
                cache_key = future_to_query[future]
                cache[cache_key] = future.result()
        except KeyboardInterrupt:
            interrupted = True
            # Cancel queued rewrites so Ctrl+C returns promptly while preserving finished work.
            pool.shutdown(wait=False, cancel_futures=True)
        else:
            pool.shutdown(wait=True)
    else:
        try:
            for cache_key, query in uncached:
                cache[cache_key] = _rewrite(query)
        except KeyboardInterrupt:
            interrupted = True

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        print(f"Warning: Could not write cache to {cache_path}: {exc}")

    if interrupted:
        print("\nBM25 rewrite precompute interrupted. Saved partial cache.")
        raise KeyboardInterrupt

    return cache

def short_string(s: str, max_length: int) -> str:
    if not s:
        return ""
    if len(s) <= max_length:
        return s
    
    return s[:max_length] + "..."
