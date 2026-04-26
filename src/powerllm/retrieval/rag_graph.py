import json
import sys
from typing import Any, Literal, cast

from langchain_core.messages import ToolMessage
from typing_extensions import TypedDict
from langchain_core.documents import Document

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing import TypedDict, Required, NotRequired

from powerllm.models.model_resolver import get_chat_model, resolve_chat_model_name
from powerllm.retrieval.pipeline import (
    get_bm25_retrieval_tool,
    get_similarity_retrieval_tool,
    reciprocal_rank_fusion,
)
from powerllm.retrieval.query_builder import QueryBuilder
from powerllm.retrieval.reranker import rerank_documents
from enum import Enum

SUPPORTED_RETRIEVAL_MODES = {"similarity", "bm25", "hybrid"}
RUN_MODES = {"full", "retrieval_only"}
RunMode = Literal["full", "retrieval_only"]

class RAGInputs(TypedDict):
    case_id: Required[str]
    query: Required[str]
    embedding_model_id: Required[str]
    chat_model_id: Required[str]
    run_mode: Required[RunMode] 
    retrieval_mode: Required[str]
    similarity_k: Required[int]
    bm25_k: Required[int]
    final_k: Required[int]
    rrf_k: NotRequired[int]
    use_reranker: NotRequired[bool]


class RAGWorking(TypedDict, total=False):
    bm25_query: NotRequired[str]
    bm25_query_rewrite_applied: NotRequired[bool]
    bm25_query_rewrite_error: NotRequired[str]
    candidate_docs: NotRequired[list[Any]]
    retrieved_docs: NotRequired[list[Any]]
    context_text: NotRequired[str]
    model_name: NotRequired[str]
    generation_prompt: NotRequired[str]
    retrieval_error: NotRequired[str]
    generation_error: NotRequired[str]


class RAGOutputs(TypedDict, total=False):
    answer: NotRequired[str]
    citations: NotRequired[list[dict[str, Any]]]
    graph_meta: NotRequired[dict[str, Any]]


class RAGState(RAGInputs, RAGWorking, RAGOutputs):
    pass


SYSTEM_PROMPT = (
    # Role
    "You are a factual legal assistant. "
    # Task
    "Answer only from the provided context. "
    "Do not use outside knowledge. "
    "If the context is insufficient, answer exactly: I don't have sufficient information to answer this question. "
    "If given a sub-clause, do not assume information from the main clause that is not explicitly stated in the sub-clause. "
    "Do not invent sources. "
    # Output Format
    "When the context contains enough information, provide a complete answer rather than an ultra-short label. "
    "Do not mention the context explicitly. "
    "Do not use meta phrases such as 'according to the context' or 'based on the provided information'. "
    "Write the answer as if directly explaining the situation."
    "If the user asks about a dispute, issue, claim, allegation, reason, background, or current situation, explain it in 2-5 sentences using the context. " # Contextual Information
    "Where applicable, include: the nature of the dispute, the current status, the key claim or issue, and any stated consequence or requested relief. "
    "Use the same language as the user's question unless the user explicitly asks for another language. "
    "If the question is in Chinese, answer in Chinese. If the question is in English, answer in English. " # Language Specification
    "Prefer a concise explanation over a one-phrase answer."
)

DEFAULT_NO_ANSWER = "I don't have sufficient information to answer this question."

class RetrievalMode(str, Enum):
    similarity = "similarity"
    bm25 = "bm25"
    hybrid = "hybrid"



def _doc_source(doc: Any) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return (
        metadata.get("source_file_name")
        or metadata.get("original_file")
        or metadata.get("source")
        or "unknown"
    )


def _doc_title(doc: Any, source: str) -> str:
    metadata = getattr(doc, "metadata", {}) or {}
    return (
        metadata.get("title")
        or metadata.get("section")
        or metadata.get("source_file_name")
        or metadata.get("original_file")
        or metadata.get("source")
        or source
    )


def _llm_content_to_text(content: Any) -> str:
    # Some providers stream structured content parts instead of raw strings.
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
          if isinstance(item, str):
              parts.append(item)
              continue
          if not isinstance(item, dict):
              continue

          item_type = item.get("type")
          if item_type == "text":
              text = item.get("text")
              if isinstance(text, str) and text:
                  parts.append(text)
        return "".join(parts)

    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text

    return ""


def _is_generate_message(metadata: Any) -> bool:
    if not isinstance(metadata, dict):
        return False

    return metadata.get("langgraph_node") == "generate"


def _build_rag_state(
    *,
    case_id: str,
    query: str,
    embedding_model_id: str,
    chat_model_id: str,
    run_mode: RunMode,
    retrieval_mode: str,
    similarity_k: int,
    bm25_k: int,
    final_k: int,
    rrf_k: int | None = None,
    use_reranker: bool = True,
) -> RAGState:
    state: RAGState = {
        "case_id": case_id,
        "query": query,
        "embedding_model_id": embedding_model_id,
        "chat_model_id": chat_model_id,
        "run_mode": run_mode,
        "retrieval_mode": retrieval_mode,
        "similarity_k": similarity_k,
        "bm25_k": bm25_k,
        "final_k": final_k,
        "use_reranker": use_reranker,
    }
    # Keep optional tuning-only retrieval knobs in the shared graph state
    # so benchmark and production callers can use the same execution path.
    if rrf_k is not None:
        state["rrf_k"] = rrf_k
    return state


def _build_rag_config(
    state: RAGState,
    *,
    run_name: str,
    tags: list[str],
    vector_store: Any | None = None,
    documents: list[Document] | None = None,
    system_prompt: str | None = None,
) -> RunnableConfig:
    return RunnableConfig(
        run_name=run_name,
        tags=tags,
        metadata={
            "case_id": state["case_id"],
            "embedding_model_id": state["embedding_model_id"],
            "chat_model_id": state["chat_model_id"],
            "run_mode": state["run_mode"],
            "retrieval_mode": state["retrieval_mode"],
            "similarity_k": state["similarity_k"],
            "bm25_k": state["bm25_k"],
            "final_k": state["final_k"],
        },
        configurable={
            "vector_store": vector_store,
            "documents": documents,
            "system_prompt": system_prompt,
        },
    )


def _graph_meta_from_state(state: RAGState) -> dict[str, Any]:
    graph_meta = {
        "model": state.get("model_name", resolve_chat_model_name(state["chat_model_id"])),
        "run_mode": state["run_mode"],
        "retrieval_mode": state["retrieval_mode"],
        "retrieval_k": state["final_k"],
        "similarity_k": state["similarity_k"],
        "bm25_k": state["bm25_k"],
        "final_k": state["final_k"],
    }
    if state.get("retrieval_error"):
        graph_meta["retrieval_error"] = state["retrieval_error"]
    if state.get("generation_error"):
        graph_meta["generation_error"] = state["generation_error"]
    return graph_meta


def _merge_graph_update(state: RAGState, update: dict[str, Any]) -> None:
    for value in update.values():
        if isinstance(value, dict):
            state.update(value)


def _invoke_rag_graph(
    state: RAGState,
    *,
    vector_store: Any | None = None,
    documents: list[Document] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    config = _build_rag_config(
        state,
        run_name="rag_query",
        tags=["rag", "sync"],
        vector_store=vector_store,
        documents=documents,
        system_prompt=system_prompt,
    )
    return _RAG_APP.invoke(state, config=config)


def _get_configurable(config: RunnableConfig | None, key: str) -> Any:
    if not isinstance(config, dict):
        return None
    configurable = config.get("configurable")
    if not isinstance(configurable, dict):
        return None
    return configurable.get(key)


def _get_run_mode_from_state(state: RAGState) -> RunMode:
    # Supported run modes:
    # - "full": execute retrieval and answer generation.
    # - "retrieval_only": stop at retrieval-oriented graph outputs and skip LLM generation.
    run_mode = _require_state_field(state, "run_mode")
    if run_mode in RUN_MODES:
        return cast(RunMode, run_mode)
    raise ValueError(f"Unsupported or missing run_mode: {run_mode!r}")


def _require_state_field(state: RAGState, field_name: str) -> Any:
    # Fail fast when a required graph input or upstream node output is missing.
    if field_name not in state:
        raise KeyError(f"RAG state is missing required field: {field_name}")
    return state[field_name]


def _retrieve_similarity_docs(
        *,
        vector_store: Any,
        query: str,
        k: int,
        ) -> list[Document]:
    """
    Retrieve relevant documents using similarity search based on vector embeddings.
    """
    similarity_tool = get_similarity_retrieval_tool(vector_store, k=k)
    tool_result = similarity_tool.invoke(
        {
            "type": "tool_call",
            "name": similarity_tool.name,
            "args": {"query": query},
            "id": "rag_similarity_retrieval",
        }
    )
    if not isinstance(tool_result, ToolMessage):
        raise TypeError(
            f"Expected ToolMessage from similarity tool, got {type(tool_result)!r}"
        )
    return tool_result.artifact or []


def _build_bm25_query(
    *,
    original_query: str, # The raw original user query
    chat_model_id: str, # We use the chat model to generate BM25-friendly keywords from the original query
    k: int, # The number of keywords to generate, between 1 and 10
) -> tuple[str, bool, str | None]:
    """
    Build a BM25-friendly query string from the original user query using chat model.
    Falls back to the original query if keyword generation fails.
    """
    model = get_chat_model(chat_model_id)
    query_builder = QueryBuilder(vector_store=None, chat_model=model)

    try:
        keyword_limit = min(max(k, 1), 10)
        rewritten = query_builder.gen_keywords_str(
            original_query,
            limit=keyword_limit,
            include_query=False,
        )
        rewritten = " ".join(rewritten.split()).strip()
        if not rewritten:
            return (
                original_query,
                False,
                "Keyword rewrite produced an empty query; falling back to original query.",
            )

        normalized_original = " ".join(original_query.split()).strip().lower()
        rewrite_applied = rewritten.lower() != normalized_original
        return rewritten, rewrite_applied, None
    except Exception as exc:
        return (
            original_query,
            False,
            f"{type(exc).__name__}: {exc}",
        ) # return original query if any error occurs during keyword generation
    

def build_bm25_query_node(state: RAGState) -> dict[str, Any]: 
    """
    Build a BM25 query and add it to the state for downstream retrieval nodes.
    """
    query = state["query"]
    retrieval_mode = state["retrieval_mode"]
    cached_bm25_query = state.get("bm25_query")

    # Reuse externally provided rewrite result when available.
    if isinstance(cached_bm25_query, str) and cached_bm25_query.strip():
        normalized_original = " ".join(query.split()).strip().lower()
        normalized_cached = " ".join(cached_bm25_query.split()).strip().lower()
        return {
            "bm25_query": cached_bm25_query,
            "bm25_query_rewrite_applied": normalized_cached != normalized_original,
        }

    # Skip rewrite query if not using BM25 retrieval
    if retrieval_mode not in {"bm25", "hybrid"}:
        return {"bm25_query": query, "bm25_query_rewrite_applied": False}

    bm25_query, rewrite_applied, rewrite_error = _build_bm25_query(
        original_query=query,
        chat_model_id=state["chat_model_id"],
        k=state["bm25_k"],
    )
    payload: dict[str, Any] = {
        "bm25_query": bm25_query,
        "bm25_query_rewrite_applied": rewrite_applied,
    }
    if rewrite_error:
        payload["bm25_query_rewrite_error"] = rewrite_error
    return payload


def _retrieve_bm25_docs(
        *,
        documents: list[Document],
        bm25_query: str, # Auto determined by previous node, could be the original query or a rewritten keyword-enhanced query depending on retrieval mode.
        k: int, # Target number of documents to retrieve, used for both BM25 and similarity retrieval to keep results comparable.
        ) -> list[Document]:
    """
    Retrieve relevant documents using BM25 retrieval.
    """
    bm25_tool = get_bm25_retrieval_tool(documents, k=k)
    tool_result = bm25_tool.invoke(
        {
            "type": "tool_call",
            "name": bm25_tool.name,
            "args": {"query": bm25_query},
            "id": "rag_bm25_retrieval",
        }
    )
    if not isinstance(tool_result, ToolMessage):
        raise TypeError(f"Expected ToolMessage from BM25 tool, got {type(tool_result)!r}")
    return tool_result.artifact or []


def retrieve_node(state: RAGState, config: RunnableConfig | None = None) -> dict[str, Any]:
    retrieval_mode = _require_state_field(state, "retrieval_mode")
    vector_store = _get_configurable(config, "vector_store")
    documents = _get_configurable(config, "documents")
    use_reranker = state.get("use_reranker", True)
    try:
        match retrieval_mode:
            case "bm25":
                # BM25 retrieval demands documents to search over.

                if not isinstance(documents, list):
                    raise ValueError("Missing documents in RunnableConfig.configurable")
                candidate_docs = _retrieve_bm25_docs(
                    documents=documents,
                    bm25_query=_require_state_field(state, "bm25_query"),
                    k=state["bm25_k"] if use_reranker else state["final_k"],
                )

                if use_reranker:
                    docs = rerank_documents(
                        query=state["query"],
                        documents=candidate_docs,
                        limit=state["final_k"],
                    )
                else:
                    docs = candidate_docs

                return {"candidate_docs": candidate_docs, "retrieved_docs": docs}
            case "similarity":
                # Similarity retrieval demands a vector store.
                if vector_store is None:
                    raise ValueError("Missing vector_store in RunnableConfig.configurable")

                candidate_docs = _retrieve_similarity_docs(
                    vector_store=vector_store,
                    query=state["query"],
                    k=state["similarity_k"] if use_reranker else state["final_k"],
                )

                if use_reranker:
                    docs = rerank_documents(
                        query=state["query"],
                        documents=candidate_docs,
                        limit=state["final_k"],
                    )
                else:
                    docs = candidate_docs

                return {"candidate_docs": candidate_docs, "retrieved_docs": docs}
            case "hybrid":
                # Hybrid mode demands both vector_store and documents.
                if vector_store is None:
                    raise ValueError("Missing vector_store in RunnableConfig.configurable")
                if not isinstance(documents, list):
                    raise ValueError("Missing documents in RunnableConfig.configurable")
                
                similarity_docs = _retrieve_similarity_docs(
                    vector_store=vector_store,
                    query=state["query"],
                    k=state["similarity_k"],
                )

                bm25_docs = _retrieve_bm25_docs(
                    documents=documents,
                    bm25_query=_require_state_field(state, "bm25_query"),
                    k=state["bm25_k"],
                )

                candidate_docs = reciprocal_rank_fusion(
                    [similarity_docs, bm25_docs],
                    k=state.get("rrf_k", 60),
                    limit=None if use_reranker else state["final_k"],
                    weights=[1.0, 1.0],
                )

                if use_reranker:
                    docs = rerank_documents(
                        query=state["query"],
                        documents=candidate_docs,
                        limit=state["final_k"],
                    )
                else:
                    docs = candidate_docs

                return {"candidate_docs": candidate_docs, "retrieved_docs": docs}
            case _:
                raise ValueError(
                    f"Unsupported retrieval_mode: {retrieval_mode}. "
                    f"Supported modes: similarity, bm25, hybrid."
                )
    except Exception as exc:
        return {
            "candidate_docs": [],
            "retrieved_docs": [],
            "retrieval_error": str(exc),
        }


def _format_context(docs: list[Document]) -> str:
    """
    Format retrieved documents into a single context string for generation.
    """
    if not docs:
        return "" # Return empty context if no documents retrieved

    parts: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(
            f"[{idx}] source={_doc_source(doc)}\n"
            f"metadata={getattr(doc, 'metadata', {})}\n"
            f"content={getattr(doc, 'page_content', '')}"
        )

    return "\n\n".join(parts)


def format_context_node(state: RAGState) -> dict[str, Any]:
    """
    Format the retrieved documents into a flat string
    """
    docs = _require_state_field(state, "retrieved_docs")
    return {"context_text": _format_context(docs)}


def build_generation_prompt_node(state: RAGState) -> dict[str, Any]:
    prompt = (
        f"Question:\\n{state['query']}\\n\\n"
        f"Context:\\n{_require_state_field(state, 'context_text')}\\n\\n"
        "Provide a concise answer grounded in the context."
    )
    return {"generation_prompt": prompt}


def route_after_format_context(state: RAGState) -> str:
    # "retrieval_only" skips the generation branch entirely and goes straight
    # to post-processing so tuning runs do not invoke the chat model.
    if _get_run_mode_from_state(state) == "retrieval_only":
        return "postprocess"
    return "build_generation_prompt"


def generate_answer_node(
    state: RAGState,
    config: RunnableConfig | None = None,
) -> dict[str, Any]:
    docs = _require_state_field(state, "retrieved_docs")
    if not docs:
        return {"answer": DEFAULT_NO_ANSWER}

    try:
        model = get_chat_model(state["chat_model_id"])
        system_prompt = _get_configurable(config, "system_prompt") or SYSTEM_PROMPT
        response_chunks: list[str] = []
        merged_chunk: Any | None = None
        for chunk in model.stream([
            ("system", system_prompt),
            ("human", _require_state_field(state, "generation_prompt")),
        ]):
            merged_chunk = chunk if merged_chunk is None else merged_chunk + chunk
            text = _llm_content_to_text(getattr(chunk, "content", ""))
            if text:
                response_chunks.append(text)
        answer = "".join(response_chunks)
        if not answer and merged_chunk is not None:
            answer = _llm_content_to_text(getattr(merged_chunk, "content", ""))
        return {"answer": answer.strip() or DEFAULT_NO_ANSWER}
    except Exception as exc:
        return {
            "answer": DEFAULT_NO_ANSWER,
            "generation_error": str(exc),
        }

def postprocess_node(state: RAGState) -> dict[str, Any]:
    """
    Post-process the retrieved documents to extract unique citations for the final output.
    The message part will NOT be included in this node.
    """

    docs = _require_state_field(state, "retrieved_docs")

    seen = set()
    citations: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        metadata = dict(getattr(doc, "metadata", {}) or {})
        source = _doc_source(doc)
        key = (source, json.dumps(metadata, sort_keys=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "id": f"citation-{idx}",
                "source": source,
                "title": _doc_title(doc, source),
                "metadata": metadata,
            }
        )

    payload: dict[str, Any] = {
        "citations": citations,
        "graph_meta": _graph_meta_from_state(state),
    }
    # Retrieval-only runs skip generation, so keep the output shape stable by
    # providing the standard fallback answer here.
    if _get_run_mode_from_state(state) == "retrieval_only" and not state.get("answer"):
        payload["answer"] = DEFAULT_NO_ANSWER
    return payload

def build_rag_graph():
    graph = StateGraph(RAGState)
    graph.add_node("build_bm25_query", build_bm25_query_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("format_context", format_context_node)
    graph.add_node("build_generation_prompt", build_generation_prompt_node)
    graph.add_node("generate", generate_answer_node)
    graph.add_node("postprocess", postprocess_node)

    graph.add_edge(START, "build_bm25_query")
    graph.add_edge("build_bm25_query", "retrieve")
    graph.add_edge("retrieve", "format_context")
    graph.add_conditional_edges(
        "format_context",
        route_after_format_context,
        {
            "build_generation_prompt": "build_generation_prompt",
            "postprocess": "postprocess",
        },
    )
    graph.add_edge("build_generation_prompt", "generate")
    graph.add_edge("generate", "postprocess")
    graph.add_edge("postprocess", END)
    return graph.compile()


_RAG_APP = build_rag_graph()


def run_rag_query_with_resources(
    *,
    case_id: str,
    query: str,
    embedding_model_id: str,
    chat_model_id: str,
    vector_store: Any | None = None,
    documents: list[Document] | None = None,
    retrieval_mode: str = "similarity",
    similarity_k: int,
    bm25_k: int,
    final_k: int,
    rrf_k: int | None = None,
    system_prompt: str | None = None,
    bm25_query: str | None = None,
    run_mode: RunMode,
    use_reranker: bool = True,
) -> dict[str, Any]:
    # Allow external callers such as benchmark/tuning to reuse the production
    # graph while injecting their own prebuilt retrieval resources.
    state = _build_rag_state(
        case_id=case_id,
        query=query,
        embedding_model_id=embedding_model_id,
        chat_model_id=chat_model_id,
        run_mode=run_mode,
        retrieval_mode=retrieval_mode,
        similarity_k=similarity_k,
        bm25_k=bm25_k,
        final_k=final_k,
        rrf_k=rrf_k,
        use_reranker=use_reranker,
    )
    # Reuse a cached BM25 rewrite when the caller already computed it.
    if isinstance(bm25_query, str) and bm25_query.strip():
        state["bm25_query"] = bm25_query
    return _invoke_rag_graph(
        state,
        vector_store=vector_store,
        documents=documents,
        system_prompt=system_prompt,
    )
