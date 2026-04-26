import hashlib

import requests
from langchain.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
import os

# Filled by configuration.Configuration when custom embedding profiles are loaded.
_custom_embedding_models: dict[str, dict[str, str]] = {}

# For using embedding models from Cloudflare
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
if not CLOUDFLARE_ACCOUNT_ID:
    raise ValueError("CLOUDFLARE_ACCOUNT_ID is not set")
CLOUDFLARE_AUTH_TOKEN = os.getenv("CLOUDFLARE_AUTH_TOKEN")
if not CLOUDFLARE_AUTH_TOKEN:
    raise ValueError("CLOUDFLARE_AUTH_TOKEN is not set")


def register_custom_embedding_models(specs: dict[str, dict[str, str]]) -> None:
    """Replace the in-memory custom embedding model specs (persisted in config.json)."""
    global _custom_embedding_models
    _custom_embedding_models = dict(specs)


EMBEDDING_MODEL_REGISTRY = {
    "gemini-embedding-001": {
        "display_name": "Gemini Embedding 1",
        "provider": "google_genai",
        "model_name": "models/gemini-embedding-001",
    },
    "gemini-embedding-2-preview": {
        "display_name": "Gemini Embedding 2 Preview",
        "provider": "google_genai",
        "model_name": "models/gemini-embedding-2-preview",
    },
    "embeddinggemma-300m": {
        "display_name": "EmbeddingGemma (LM Studio)",
        "provider": "openai_compatible",
        "model_name": "text-embedding-embeddinggemma-300m-qat (LM Studio)",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "lm-studio",
    },
    "Qwen3-Embedding-0.6B-4bit-DWQ": {
        "display_name": "Qwen3 Embedding 0.6B 4bit DWQ (Omlx)",
        "provider": "openai_compatible",
        "model_name": "Qwen3-Embedding-0.6B-4bit-DWQ",
        "base_url": "http://127.0.0.1:1234/v1",
        "api_key": "Bearer omlx",
    },
    "Qwen3-Embedding-0.6B-4bit-DWQ-Online": {
        "display_name": "Qwen3 Embedding 0.6B 4bit DWQ (Online)",
        "provider": "openai_compatible",
        "model_name": "@cf/qwen/qwen3-embedding-0.6b",
        "base_url": f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
        "api_key": CLOUDFLARE_AUTH_TOKEN,
    },
}

# Reachability-only probe for Google GenAI (registry has no per-model HTTP base).
_GOOGLE_GENAI_PING_URL = "https://generativelanguage.googleapis.com/"
_HTTP_CHECK_TIMEOUT_S = 5.0


def check_embedding_model_connection(embedding_model_id: str) -> bool:
    """
    Quick HTTP reachability check. Does not validate API keys or that the embedding model exists.
    """
    spec = EMBEDDING_MODEL_REGISTRY.get(embedding_model_id)
    custom = _custom_embedding_models.get(embedding_model_id)
    if custom:
        try:
            requests.get(custom["base_url"], timeout=_HTTP_CHECK_TIMEOUT_S)
        except requests.exceptions.RequestException:
            return False
        return True
    if not spec:
        raise ValueError(f"Invalid embedding model id: {embedding_model_id}")

    try:
        if spec["provider"] == "google_genai":
            requests.get(_GOOGLE_GENAI_PING_URL, timeout=_HTTP_CHECK_TIMEOUT_S)
        elif spec["provider"] == "openai_compatible":
            requests.get(spec["base_url"], timeout=_HTTP_CHECK_TIMEOUT_S)
        else:
            raise ValueError(f"Unsupported embedding provider: {spec['provider']}")
    except requests.exceptions.RequestException:
        return False
    return True


def get_available_embedding_models() -> list[dict[str, str]]:
    return [
        {"id": model_id, "name": spec["display_name"]}
        for model_id, spec in EMBEDDING_MODEL_REGISTRY.items()
    ]


def _embedding_api_key_secret(spec: dict[str, str]) -> SecretStr:
    key = (spec.get("api_key") or "").strip()
    if key:
        return SecretStr(key)
    return SecretStr("Bearer omlx")


def _openai_compatible_embeddings(
    model_name: str,
    base_url: str,
    api_key: SecretStr,
) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        chunk_size=10,
        show_progress_bar=False,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )


def get_embeddings_model(embedding_model_id: str):
    custom = _custom_embedding_models.get(embedding_model_id)
    # All custom embedding models are openai compatible
    if custom:
        print(f"Using {custom['display_name']} (custom).")
        return _openai_compatible_embeddings(
            custom["model_name"],
            custom["base_url"],
            _embedding_api_key_secret(custom),
        )

    spec = EMBEDDING_MODEL_REGISTRY.get(embedding_model_id)
    if not spec:
        raise ValueError(f"Invalid embedding model id: {embedding_model_id}")
    
    if spec["provider"] == "google_genai":
        print(f"Using {spec['display_name']}.")
        return GoogleGenerativeAIEmbeddings(model=spec["model_name"])

    if spec["provider"] == "openai_compatible":
        print(f"Using {spec['display_name']}.")
        return _openai_compatible_embeddings(
            spec["model_name"],
            spec["base_url"],
            SecretStr(spec.get("api_key", "lm-studio")),
        )

    raise ValueError(f"Unsupported embedding provider: {spec['provider']}")


def _serialize_retrieved_docs(retrieved_docs: list[Document]) -> str:
    """
    Convert retrieved documents into the text/artifact payload expected by tools.
    :param retrieved_docs: List of Document objects retrieved by the retriever.
    """
    return "\n\n".join(
        f"Source: {doc.metadata}\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )


def _document_fusion_key(doc: Document) -> str:
    """
    Build a stable key for cross-retriever deduplication during fusion.
    """
    metadata = doc.metadata or {}
    if metadata.get("chunk_id"):
        return str(metadata["chunk_id"])
    source = metadata.get("original_file") or metadata.get("source") or "unknown"
    return f"{source}::{doc.page_content}"


def build_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    *,
    k: int = 60,
    limit: int | None = None,
    weights: list[float],
) -> list[Document]:
    """
    Fuse multiple ranked retrieval results with Reciprocal Rank Fusion (RRF).

    :param ranked_lists: Each inner list is an ordered retrieval result set.
    :param k: RRF rank constant. Larger values flatten rank differences.
    :param limit: Optional cap for the number of returned documents.
    :param weights: Per-list weights for weighted RRF.
        - 1. weights for similarity
        - 2. weights for bm25
    """
    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: dict[str, float] = {}
    best_docs: dict[str, Document] = {}

    for ranked_docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(ranked_docs, start=1):
            key = _document_fusion_key(doc)
            scores[key] = scores.get(key, 0.0) + (weight / (k + rank))
            best_docs.setdefault(key, doc)

    fused_docs = [
        best_docs[key]
        for key, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]

    if limit is not None:
        return fused_docs[:limit]
    return fused_docs


def get_similarity_retrieval_tool(vector_store, k: int = 5):
    """
    Create a tool that only performs dense vector similarity search.

    This tool is best for semantic retrieval, where the user's wording may not
    exactly match the source text but the meaning is similar. Prefer it for
    broad questions, paraphrased concepts, summaries, relationships, and other
    cases where contextual meaning matters more than exact keywords.

    The input should be a concise search query describing the information need.
    :param vector_store: An already initialized vector store with the documents indexed. ChromaDB is used now.
    :param k: The number of similar documents to retrieve.
    """
    @tool(response_format="content_and_artifact")
    def retrieve_context_by_similarity(query: str):
        """
        Retrieve context with vector similarity search.

        Use this when semantic meaning matters more than exact term matching.
        It is useful for paraphrased questions, high-level concepts, summaries,
        and relation-seeking queries where relevant text may use different
        wording from the user query.
        """
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)
        serialized = _serialize_retrieved_docs(retrieved_docs)
        return serialized, retrieved_docs

    return retrieve_context_by_similarity


def get_bm25_retrieval_tool(documents: list[Document], k: int = 5):
    """
    Create a tool that only performs sparse BM25 search over an existing corpus.

    This tool is best for exact or near-exact keyword retrieval. Prefer it when
    the query depends on names, identifiers, dates, organizations, addresses,
    legal terms, or other wording that is likely to appear literally in the
    source text.

    The input should be a concise keyword-focused query rather than a long
    natural-language paragraph.
    """
    searchable_docs = [doc for doc in documents if doc.page_content.strip()]

    @tool(response_format="content_and_artifact")
    def retrieve_context_by_bm25(query: str):
        """
        Retrieve context with BM25 keyword search.

        Use this when exact term matching matters, especially for names, case
        numbers, dates, terminology, and other precise strings that should be
        found directly in the source text.
        """
        if not searchable_docs:
            return "", []

        # Build the retriever from the provided chunk corpus so BM25 can rank keyword matches.
        retriever = BM25Retriever.from_documents(searchable_docs)
        retriever.k = k
        retrieved_docs = retriever.invoke(query)
        serialized = _serialize_retrieved_docs(retrieved_docs)
        return serialized, retrieved_docs

    return retrieve_context_by_bm25


class Pipeline:
    def __init__(self):
        # Any other extensions are not currently supported
        self.direct_extensions = [".md", ".txt", ".html"]
        self.pre_process_extensions = [".pdf"]
