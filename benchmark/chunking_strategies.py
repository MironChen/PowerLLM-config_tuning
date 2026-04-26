from __future__ import annotations

from typing import Literal

from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import powerllm.retrieval.chunker as chunker


ChunkingStrategy = Literal["legal", "langchain_basic"]
DEFAULT_CHUNKING_STRATEGY: ChunkingStrategy = "legal"


def build_benchmark_chunks(
    documents: list[Document],
    *,
    strategy: ChunkingStrategy = DEFAULT_CHUNKING_STRATEGY,
    chunking_config: chunker.ChunkingConfig | None = None,
) -> list[Document]:
    """Build benchmark chunks using the selected strategy.

    `legal` reuses the production retrieval chunker. `langchain_basic` is a
    plain RecursiveCharacterTextSplitter baseline with no legal-specific
    separators, prefix merging, or section/header injection.
    """
    if not documents:
        raise ValueError("No documents to build retrieval chunks from.")

    if strategy == "legal":
        return chunker.build_retrieval_chunks(
            documents,
            chunking_config=chunking_config,
        )

    if strategy == "langchain_basic":
        config = chunking_config or chunker.ChunkingConfig()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        return filter_complex_metadata(splitter.split_documents(documents))

    raise ValueError(f"Unsupported chunking strategy: {strategy}")
