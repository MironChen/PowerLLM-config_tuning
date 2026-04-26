from __future__ import annotations

from typing import Any

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

import powerllm.retrieval.chunker as chunker
from benchmark.benchmark_utils import (
    annotate_documents,
    build_benchmark_cache_dir,
    build_benchmark_collection_name,
    serialize_retrieved_docs,
)
from benchmark.chunking_strategies import (
    DEFAULT_CHUNKING_STRATEGY,
    ChunkingStrategy,
    build_benchmark_chunks,
)
from powerllm.retrieval.rag_graph import RunMode, run_rag_query_with_resources


def _require_result_field(result: dict[str, Any], field_name: str) -> Any:
    # Fail fast when the shared graph stops returning a required field.
    if field_name not in result:
        raise KeyError(f"Shared RAG graph result is missing required field: {field_name}")
    return result[field_name]


def get_or_create_persistent_vector_store(
    *,
    title: str,
    context: str,
    embedding_model: Any,
    embedding_model_id: str,
    cache_root,
    chunking_config: chunker.ChunkingConfig | None = None,
    chunking_strategy: ChunkingStrategy = DEFAULT_CHUNKING_STRATEGY,
) -> dict[str, Any]:
    documents = build_benchmark_chunks(
        [
            Document(page_content=context, metadata={"title": title, "source": title}),
        ],
        strategy=chunking_strategy,
        chunking_config=chunking_config,
    )
    documents = annotate_documents(
        title=title,
        context=context,
        documents=documents,
    )

    collection_name = build_benchmark_collection_name(
        title=title,
        context=context,
        chunking_strategy=chunking_strategy,
    )
    persist_directory = build_benchmark_cache_dir(
        title=title,
        context=context,
        embedding_model_id=embedding_model_id,
        cache_root=cache_root,
        chunking_config=chunking_config,
        chunking_strategy=chunking_strategy,
    )
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(persist_directory),
        client_settings=Settings(anonymized_telemetry=False),
    )

    existing_count = vector_store._collection.count()
    if existing_count == 0:
        vector_store.add_documents(documents)

    return {
        "vector_store": vector_store,
        "documents": documents,
        "persist_directory": str(persist_directory),
        "cache_hit": existing_count > 0,
    }


def release_persistent_vector_store(embedded_database: Any) -> None:
    """Release Chroma resources opened for a benchmark vector store."""
    if not isinstance(embedded_database, dict):
        return

    vector_store = embedded_database.get("vector_store")
    client = getattr(vector_store, "_client", None)
    clear_system_cache = getattr(client, "clear_system_cache", None)
    if callable(clear_system_cache):
        clear_system_cache()

    embedded_database.pop("vector_store", None)
    embedded_database.pop("documents", None)


def get_answer(
    *,
    query: str,
    embedded_database: Any,
    chat_model_id: str,
    system_prompt: str,
    retrieval_mode: str = "similarity",
    similarity_k: int = 5,
    bm25_k: int = 5,
    final_k: int = 5,
    rrf_k: int = 60,
    bm25_query_override: str | None = None,
    run_mode: RunMode,
    use_reranker: bool = False,
) -> dict[str, Any]:
    # The benchmark now executes the shared production graph and only injects
    # benchmark-managed resources and overrides from this wrapper layer.
    result = run_rag_query_with_resources(
        case_id="benchmark",
        query=query,
        embedding_model_id="benchmark",
        chat_model_id=chat_model_id,
        vector_store=embedded_database.get("vector_store") if isinstance(embedded_database, dict) else None,
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
        system_prompt=system_prompt,
        bm25_query=bm25_query_override,
        run_mode=run_mode,
        use_reranker=use_reranker,
    )
    candidate_docs = _require_result_field(result, "candidate_docs")
    retrieved_docs = _require_result_field(result, "retrieved_docs")
    response: dict[str, Any] = {
        "retrieved_chunks": serialize_retrieved_docs(candidate_docs),
        "final_retrieved_chunks": serialize_retrieved_docs(retrieved_docs),
        "bm25_query": result.get("bm25_query"),
        "bm25_query_rewrite_applied": result.get("bm25_query_rewrite_applied", False),
        "bm25_query_rewrite_error": result.get("bm25_query_rewrite_error"),
    }
    if run_mode == "full":
        response["predicted_answer"] = _require_result_field(result, "answer")
    return response
