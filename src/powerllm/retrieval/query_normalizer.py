# Translation table for converting full-width digits to half-width digits
import unicodedata
import re


FULLWIDTH_DIGITS = str.maketrans(
    "０１２３４５６７８９",
    "0123456789"
)

# Common punctuation normalization map
# CN punctuation -> ASCII punctuation (EN)
PUNCT_NORMALIZATION = {
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "《": "<",
    "》": ">",
    "，": ",",
    "。": ".",
    "：": ":",
    "；": ";",
    "！": "!",
    "？": "?",
}

class QueryNormalizer:

    def _clean_query(self, text: str) -> str:
        """
        Validate query type, then normalize whitespace.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("text cannot be empty")

        return cleaned

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to NFKC form.

        This converts compatibility characters into standard forms.
        Example:
            full-width Latin letters -> ASCII
            full-width numbers -> ASCII
        """
        return unicodedata.normalize("NFKC", text)

    def _normalize_punctuation(self, text: str) -> str:
        """
        Replace common Chinese punctuation with ASCII equivalents.
        This helps downstream regex matching and tokenization.
        """
        for src, tgt in PUNCT_NORMALIZATION.items():
            text = text.replace(src, tgt)
        return text

    def _normalize_english_citations(self, text: str) -> str:
        """
        Normalize English-style legal citations into a consistent form.

        Examples:
            Art. 18 -> Article 18
            art 18  -> Article 18
        """
        text = re.sub(r"\b[Aa]rt\.?\s*(\d+)", r"Article \1", text)
        return text

    def _collapse_whitespace(self, text: str) -> str:
        """
        Collapse multiple whitespace characters into a single space.
        """
        return re.sub(r"\s+", " ", text).strip()

    def basic_query_clean(self, query: str) -> str:
        """
        Perform lightweight preprocessing for a legal retrieval query.

        Steps:
            1. Unicode normalization (NFKC)
            2. Convert full-width digits to ASCII digits
            3. Normalize punctuation
            4. Normalize English citation forms
            5. Collapse redundant whitespace

        This function intentionally avoids modifying numbers or semantics.
        It only standardizes formatting for later parsing and normalization.
        """

        if not query:
            return ""
        
        # Step 0: Validate input type and clean whitespace
        query = self._clean_query(query)

        # Step 1: Normalize Unicode characters
        query = self._normalize_unicode(query)

        # Step 2: Convert full-width digits to ASCII digits
        query = query.translate(FULLWIDTH_DIGITS)

        # Step 3: Normalize punctuation variants
        query = self._normalize_punctuation(query)

        # Step 4: Normalize English legal citations
        query = self._normalize_english_citations(query)

        # Step 5: Collapse extra whitespace
        query = self._collapse_whitespace(query)

        return query
