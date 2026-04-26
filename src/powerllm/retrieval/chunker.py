import json
import re
from dataclasses import dataclass
from pathlib import Path

import cn2an
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# Legal-structure-aware separators, ordered from coarsest to finest
LEGAL_SEPARATORS = [
    r"\n(?:ARTICLE|Article)\s+",
    r"\n(?:SECTION|Section)\s+",
    r"\n(?:CLAUSE|Clause)\s+",
    r"\n(?:PART|Part)\s+",
    r"\n(?:CHAPTER|Chapter)\s+",
    r"\n(?:SCHEDULE|Schedule)\s+",
    r"\n\d+(?:\.\d+)+[.)]?\s+",  # numbered clauses like "1.1", "2.3.4)"
    r"\n\([A-Za-z]\)",  # lettered subclauses like "(a)", "(B)"
    r"\n\([ivxIVX]+\)",  # roman subclauses like "(i)", "(IV)"
    r"\n\d+\.",  # numbered clauses like "1.", "12." (after subclauses)
    "\n\n",
    "\n",
    ". ",  # sentence boundary before word break
    " ",
]


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 2000
    chunk_overlap: int = 500
    separators: tuple[str, ...] = tuple(LEGAL_SEPARATORS)
    is_separator_regex: bool = True


STRUCTURAL_PREFIX_MAX_CHARS = 80


def _clone_docling_provenance(provenance: list[dict] | None) -> list[dict]:
    if not provenance:
        return []

    cloned_provenance: list[dict] = []
    for item in provenance:
        cloned_provenance.append(dict(item))
    return cloned_provenance


def _merge_docling_provenance_with_prefixes(
    pending_prefixes: list[Document],
    document: Document,
) -> None:
    combined_provenance: list[dict] = []

    for prefix in pending_prefixes:
        prefix_text = prefix.page_content.strip()
        if not prefix_text:
            continue
        combined_provenance.extend(
            _clone_docling_provenance(prefix.metadata.get("docling_prov"))
        )

    combined_provenance.extend(
        _clone_docling_provenance(document.metadata.get("docling_prov"))
    )

    if combined_provenance:
        document.metadata["docling_prov"] = combined_provenance


def _normalize_document_metadata(doc: Document) -> Document:
    """
    Normalize source metadata for retrieval/display.

    Old snapshots may store an absolute raw-file path in `source`, which leaks
    internal filesystem paths into citations. Prefer the original uploaded file
    name when available, otherwise collapse `source` to its basename.

    Example:
    - Input: {"source": "/path/to/uploads/contract.pdf", "source_file_name": "contract.pdf"}
    - Output: {"source": "contract.pdf"}

    """
    metadata = dict(doc.metadata or {})
    source_file_name = metadata.get("source_file_name")
    original_file = metadata.get("original_file")
    source = metadata.get("source")

    if isinstance(source_file_name, str) and source_file_name.strip():
        metadata["source"] = Path(source_file_name).name
    elif isinstance(original_file, str) and original_file.strip():
        metadata["source"] = Path(original_file).name
    elif isinstance(source, str) and source.strip():
        metadata["source"] = Path(source).name

    doc.metadata = metadata
    return doc


def _build_text_splitter(
    chunking_config: ChunkingConfig | None = None,
) -> RecursiveCharacterTextSplitter:
    config = chunking_config or ChunkingConfig()
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=list(config.separators),
        is_separator_regex=config.is_separator_regex,
    )


def _compose_markdown_heading_context(metadata: dict | None) -> tuple[str | None, str]:
    """
    Build a stable heading path from markdown header metadata.

    Example:
    - Input: {"Header_1": "Section 1", "Header_2": "Subsection A"}
    - Output: ("Subsection A", "Section 1 > Subsection A")
    """
    if not metadata:
        return None, ""

    # Extract and clean header values in order
    headings = []
    for header_key in ("Header_1", "Header_2", "Header_3"):
        header_value = metadata.get(header_key)
        if isinstance(header_value, str):
            cleaned = header_value.strip()
            if cleaned:
                headings.append(cleaned)

    if not headings:
        return None, ""

    # Return deepest title and full breadcrumb path
    deepest_title = headings[-1]
    breadcrumb_path = " > ".join(headings)
    return deepest_title, breadcrumb_path


def _prepend_markdown_heading_context(document: Document) -> Document:
    """
    Ensure markdown heading metadata is also present in the document text.
    """
    title, heading_path = _compose_markdown_heading_context(document.metadata)
    if title and not document.metadata.get("title"):
        document.metadata["title"] = title

    if not heading_path:
        return document

    content = document.page_content.lstrip()
    if content.startswith(heading_path) or content.startswith(f"[{heading_path}]"):
        return document

    document.page_content = f"[{heading_path}]\n{content}"
    return document


def _extract_section_header(text: str) -> str | None:
    """Extract the leading section/article header from a chunk if present."""
    patterns = [
        r"^((?:ARTICLE|Article)\s+[IVXLCDM\d]+[^\n]*)",
        r"^((?:SECTION|Section)\s+[\d\.]+[^\n]*)",
        r"^((?:CLAUSE|Clause)\s+[\d\.]+[^\n]*)",
        r"^((?:PART|Part)\s+[A-ZIVXLCDM\d]+[^\n]*)",
        r"^((?:CHAPTER|Chapter)\s+[A-ZIVXLCDM\d]+[^\n]*)",
        r"^((?:SCHEDULE|Schedule)\s+[A-ZIVXLCDM\d]+[^\n]*)",
    ]
    for pattern in patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def build_retrieval_chunk_text(*, text: str, section: str | None) -> str:
    """
    Build the retrieval text for a chunk from its raw text and optional section.

    Example:
    - Input: Section 1, Text content
    - Output: [Section 1]\nText content
    """
    if section:
        return f"[{section}]\n{text}"
    return text


def get_chunk_source_text(document: Document) -> str:
    """
    Recover the original source-aligned text for a chunk.

    Example:
    - Input: [Section 1]\nText content
    - Output: Text content
    """
    text = document.page_content
    section = document.metadata.get("section")

    if section is None:
        return text
    prefix = f"[{section}]\n"

    if not text.startswith(prefix):
        raise ValueError("Chunk content does not match the stored section prefix")
    return text[len(prefix) :]


def _is_structural_prefix_text(text: str) -> bool:
    """
    Detect short standalone lines that are more likely to be headings/prefixes
    than retrieval-worthy content by themselves.
    """
    normalized = text.strip()
    if not normalized:
        return False

    if len(normalized) > STRUCTURAL_PREFIX_MAX_CHARS:
        return False

    line_count = len([line for line in normalized.splitlines() if line.strip()])
    if line_count > 2:
        return False

    # Check for Chinese legal structural markers (e.g., "第五条", "第一百二十四条")
    # Uses cn2an to validate the numeral portion
    chinese_section_match = re.match(
        r"^第([一二三四五六七八九十百千万零〇]+)[编章节条部分款]\s*$", normalized
    )
    if chinese_section_match:
        try:
            cn2an.cn2an(chinese_section_match.group(1), mode="strict")
            return True
        except ValueError:
            pass

    # Check for Chinese numbered items (e.g., "一、", "二十.")
    chinese_number_match = re.match(
        r"^([一二三四五六七八九十百千万零〇]+)[、.．]\s*$", normalized
    )
    if chinese_number_match:
        try:
            cn2an.cn2an(chinese_number_match.group(1), mode="strict")
            return True
        except ValueError:
            pass

    if normalized.endswith(("：", ":")):
        return True
    if re.match(r"^[-*•]\s*\d+\s*[.．、]?", normalized):
        return True

    return False


def _can_attach_prefix(prefix_doc: Document, next_doc: Document) -> bool:
    prefix_meta = prefix_doc.metadata or {}
    next_meta = next_doc.metadata or {}
    keys = ("source", "original_file", "source_file_name", "document_id", "case_id")
    for key in keys:
        prefix_value = prefix_meta.get(key)
        next_value = next_meta.get(key)
        if prefix_value and next_value and prefix_value != next_value:
            return False
    return True


def _merge_structural_prefix_documents(documents: list[Document]) -> list[Document]:
    """
    Merge short heading-like fragments into the following content block.

    Docling snapshots often emit items like "第五节" or "规定：" as standalone
    documents. Left untouched, they become isolated retrieval chunks that lose
    their relationship to the substantive text that follows.

    Before:
    - [Document1] "第五节" / "Section 5"
    - [Document2] "规定：" / "Rules:"
    - [Document3] "内容：" / "Content:"

    After:
    - [Document1] "第五节\n规定：\n内容："
    """
    merged: list[Document] = []
    pending_prefixes: list[Document] = []

    for document in documents:
        content = document.page_content.strip()
        if _is_structural_prefix_text(content):
            pending_prefixes.append(document)
            continue

        if pending_prefixes and all(
            _can_attach_prefix(prefix, document) for prefix in pending_prefixes
        ):
            prefix_text = "\n".join(
                prefix.page_content.strip()
                for prefix in pending_prefixes
                if prefix.page_content.strip()
            )
            if prefix_text:
                _merge_docling_provenance_with_prefixes(pending_prefixes, document)
                document.page_content = (
                    f"{prefix_text}\n{document.page_content.lstrip()}"
                )
                start_index = pending_prefixes[0].metadata.get("snapshot_item_index")
                if start_index is not None:
                    document.metadata["snapshot_item_index"] = start_index
                document.metadata["merged_prefix_count"] = len(pending_prefixes)
            pending_prefixes = []

        merged.append(document)

    merged.extend(pending_prefixes)
    return merged


def build_retrieval_chunks(
    documents: list[Document],
    chunking_config: ChunkingConfig | None = None,
) -> list[Document]:
    """
    The only function that should be called by the user to get retrieval chunks.

    :param documents: List of Document objects to build retrieval chunks from
    :param chunking_config: Chunking configuration. This is for config tuning purposes.
    :return: List of retrieval chunk Document objects
    """
    if not documents:
        raise ValueError("No documents to build retrieval chunks from.")

    normalized_documents = normalize_source_documents(documents)
    final_chunks = chunk_normalized_documents(normalized_documents, chunking_config)
    return final_chunks



def normalize_source_documents(documents: list[Document]) -> list[Document]:
    """
    Normalize source documents before text-based retrieval chunking.

    This step happens before character-based chunking. It cleans metadata and
    merges short structural fragments back into the substantive content they
    belong to.

    :param documents: List of Document objects to normalize
    :return: List of normalized Document objects
    """
    if not documents:
        return []

    normalized_documents = [_normalize_document_metadata(doc) for doc in documents]
    normalized_documents = _merge_structural_prefix_documents(normalized_documents)
    return normalized_documents


def chunk_normalized_documents(
    documents: list[Document],
    chunking_config: ChunkingConfig | None = None,
) -> list[Document]:
    """
    Split normalized source documents into character-based retrieval chunks.

    :param documents: List of normalized Document objects to chunk
    :param chunking_config: Chunking configuration. This is for config tuning purposes.
    :return: List of chunked Document objects
    """
    if not documents:
        return []

    chunks = _build_text_splitter(
        chunking_config,
    ).split_documents(documents)

    for chunk in chunks:
        section = _extract_section_header(chunk.page_content)
        if section:
            chunk.metadata["section"] = section
        chunk.page_content = build_retrieval_chunk_text(
            text=chunk.page_content,
            section=section,
        )

    return filter_complex_metadata(chunks)


def chunk_plain_text(
    text: str,
    metadata: dict | None = None,
    chunking_config: ChunkingConfig | None = None,
) -> list[Document]:
    """
    Chunk a single plain-text input into retrieval-ready Document chunks.

    :param text: The plain text to chunk
    :param metadata: Metadata to add to the Document objects
    :param chunking_config: Chunking configuration. This is for config tuning purposes.
    :return: List of retrieval chunk Document objects
    """
    if not text:
        return []

    source_documents = [
        Document(
            page_content=text,
            metadata=metadata or {},
        )
    ]
    return build_retrieval_chunks(source_documents, chunking_config)


def save_chunk_snapshot(
    file_name: str, chunks: list[Document], chunks_dir: Path
) -> Path:
    """
    Persist the final retrieval chunks exactly as they were indexed.
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    output_path = chunks_dir / f"{file_name}_chunks.json"

    serializable_chunks = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]

    tmp_path = output_path.with_name(f"{output_path.name}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=4)
            f.flush()
        tmp_path.replace(output_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return output_path


def load_chunk_snapshot(json_path: Path) -> list[Document]:
    """
    Load persisted retrieval chunks without re-running chunking.
    """
    if not json_path or not Path(json_path).exists():
        raise FileNotFoundError(f"Chunk snapshot not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_chunks = json.load(f)

    chunks: list[Document] = []
    for item in raw_chunks:
        if not isinstance(item, dict):
            continue

        page_content = item.get("page_content", "")
        metadata = item.get("metadata", {}) or {}
        if not isinstance(page_content, str):
            continue
        if not isinstance(metadata, dict):
            metadata = {}

        chunks.append(Document(page_content=page_content, metadata=dict(metadata)))

    return chunks


def load_snapshot_source_documents(json_path: Path) -> list[Document]:
    """
    Load and normalize source documents from a persisted JSON snapshot.
    """
    if not json_path or not Path(json_path).exists():
        print(f"⚠️  File not found: {json_path}")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_documents = json.load(f)

        documents = []
        for index, item in enumerate(raw_documents):
            if not isinstance(item, dict):
                continue
            page_content = item.get("page_content", "")
            if not isinstance(page_content, str):
                continue
            metadata = item.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata = dict(metadata)
            metadata.setdefault("snapshot_item_index", index)
            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata,
                )
            )

        if not documents:
            return []

        documents = normalize_source_documents(documents)
        return filter_complex_metadata(documents)
    except Exception as e:
        print(f"❌ Failed to process snapshot source JSON {json_path.name}: {e}")
        return []


def load_markdown_source_documents(markdown_file_path: Path) -> list[Document]:
    """
    Load and normalize source documents by parsing a markdown file from disk.
    """
    if not markdown_file_path or not Path(markdown_file_path).exists():
        print(f"⚠️  File not found: {markdown_file_path}")
        return []

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    try:
        if markdown_file_path.suffix.lower() != ".md":
            raise ValueError(
                f"Unsupported markdown file type: {markdown_file_path.suffix}"
            )

        with open(markdown_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split markdown file into documents by headers and prepend markdown heading context to the documents
        header_documents = header_splitter.split_text(content)
        # Prepend markdown heading context to the documents
        header_documents = [
            _prepend_markdown_heading_context(document)
            for document in header_documents
        ]
        # Normalize the documents
        normalized_documents = normalize_source_documents(header_documents)

        # Add source metadata to the documents
        for document in normalized_documents:
            document.metadata["source"] = markdown_file_path.name
            document.metadata.setdefault("snapshot_item_index", 0)
            _normalize_document_metadata(document)

        print(
            f"📝 Markdown processed: {markdown_file_path.name} ({len(normalized_documents)} normalized documents)"
        )
        return filter_complex_metadata(normalized_documents)
    except Exception as e:
        print(f"❌ Failed to process {markdown_file_path.name}: {e}")
        return []
