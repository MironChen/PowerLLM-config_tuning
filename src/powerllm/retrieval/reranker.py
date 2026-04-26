from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document

RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def _get_reranker():
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Install it with: venv/bin/python -m pip install sentence-transformers"
        ) from exc

    return CrossEncoder(RERANKER_MODEL_NAME)


def rerank_documents(
    *,
    query: str,
    documents: list[Document],
    limit: int,
) -> list[Document]:
    pairs = [(query, doc.page_content) for doc in documents]
    model = _get_reranker()
    scores = model.predict(pairs)

    ranked = sorted(
        zip(documents, scores, strict=False),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    return [doc for doc, _score in ranked[:limit]]
