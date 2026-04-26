import re
import unicodedata
from typing import Any

from langchain_core.runnables import RunnableLambda

_LEADING_WRAPPER_RE = re.compile(
    r"^\s*(?:final\s+answer|answer)\s*:\s*|^\s*the\s+answer\s+is\s+",
    re.IGNORECASE,
)
_TRAILING_NOISE_RE = re.compile(r'[\s"\'`]+$')
_BOUNDARY_QUOTES_RE = re.compile(r'^[\'"`]+|[\'"`]+$')
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:%)\]/])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[])\s+")
_WHITESPACE_RE = re.compile(r"\s+")
_SUFFIX_PATTERNS = (
    (re.compile(r"\bl\s*\.\s*l\s*\.\s*c\s*\.?(?=\W|$)", re.IGNORECASE), "llc"),
    (re.compile(r"\bl\s*\.\s*l\s*\.\s*p\s*\.?(?=\W|$)", re.IGNORECASE), "llp"),
    (re.compile(r"\binc\s*\.?(?=\W|$)", re.IGNORECASE), "inc"),
    (re.compile(r"\bcorp\s*\.?(?=\W|$)", re.IGNORECASE), "corp"),
    (re.compile(r"\bco\s*\.?(?=\W|$)", re.IGNORECASE), "co"),
)
_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


def _coerce_text(text: Any) -> str:
    return "" if text is None else str(text)


def _strip_leading_wrapper(text: str) -> str:
    previous = None
    while text != previous:
        previous = text
        text = _LEADING_WRAPPER_RE.sub("", text).strip()
        text = _BOUNDARY_QUOTES_RE.sub("", text).strip()
    return text


def _normalize_suffixes(text: str) -> str:
    for pattern, replacement in _SUFFIX_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def normalize_answer(text: str) -> str:
    """Normalize an answer string for deterministic token-level comparison."""
    normalized = _coerce_text(text)
    if not normalized.strip():
        return ""

    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.translate(_PUNCT_TRANSLATION)
    normalized = normalized.lower().strip()
    normalized = _strip_leading_wrapper(normalized)
    normalized = _normalize_suffixes(normalized)
    normalized = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", normalized)
    normalized = _SPACE_AFTER_OPEN_RE.sub(r"\1", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()
    normalized = _BOUNDARY_QUOTES_RE.sub("", normalized).strip()
    normalized = _TRAILING_NOISE_RE.sub("", normalized)
    return normalized.strip()


normalize_answer_runnable = RunnableLambda(normalize_answer)


def get_normalize_answer_runnable() -> RunnableLambda:
    """Return a LangChain runnable wrapper around normalize_answer."""
    return normalize_answer_runnable
