import powerllm.retrieval.pipeline as pipeline
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from powerllm.retrieval.query_normalizer import QueryNormalizer
import re
import cn2an

class KeywordList(BaseModel):
    keywords: list[str] = Field(
        default_factory=list,
        description="Retrieval-friendly keywords or short phrases for the user query",
    )


SECTION_PATTERNS = [
    re.compile(r"(?:第)?(\d+)(条|款|项|章|节)")
]

class QueryBuilder:
    def __init__(self, vector_store, chat_model):
        self.vector_store = vector_store
        self.chat_model = chat_model

    
    def normalize_query(self, query: str) -> str:
        """
        Normalize the query by cleaning it and converting to lowercase.
        This can help improve matching in the vector store by reducing variability.
        """

        normalized_query = QueryNormalizer().basic_query_clean(query)
        return normalized_query.lower()
    

    def _safe_an2cn_number(self, num_str: str) -> str:
        """
        Convert an Arabic numeral string to Chinese numerals.
        For example, "18" -> "十八". This is used to generate Chinese variants of legal references.
        If the input is not a valid number, return it unchanged.
        """
        return cn2an.an2cn(inputs=num_str, mode="low")
    
    
    def extract_cn_legal_forms(self, text: str) -> list[str]:
        """
        Extract legal reference numbers and return Chinese numeral forms.

        Example:
            "查找第18条和第108条" -> ["第十八条", "第一百零八条"]
        """

        variants = set()

        for pattern in SECTION_PATTERNS:
            for match in pattern.finditer(text):
                full = match.group(0)
                num = match.group(1)

                cn_num = self._safe_an2cn_number(num)
                cn_full = full.replace(num, cn_num)

                variants.add(cn_full)

        return sorted(variants)


    def gen_keywords(self, query: str, limit: int = 5) -> KeywordList:
        """
        Generate keywords from the user query to enhance retrieval.
        This can help capture the main concepts of the query and improve recall,
        especially for queries that are short or use uncommon phrasing.

        :param query: The original user query.
        :param limit: Optional maximum number of keywords to extract.
        """
        query = self.normalize_query(query)
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer")
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if limit > 20:
            raise ValueError("limit must be <= 20")

        system_prompt = (
            """
            You are an assistant that extracts retrieval keywords from user queries.

            Task:
            Extract retrieval terms for BM25.

            Rules:
            1) Return a single field `keywords`.
            2) Include exact names, dates, numbers, citations, section labels, and identifiers when present.
            3) Use concise retrieval-friendly keywords or short phrases likely to appear in source documents.
            4) Include formal or domain terms only when clearly supported by the query.
            5) Do not invent facts or broaden the meaning.
            6) Keep at most {limit} keywords.
            """).strip()

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    """
                    User query:
                    {query}
        
                    """.strip(),
                ),
            ]
        )

        structured_model = self.chat_model.with_structured_output(KeywordList)
        chain = prompt | structured_model
        # Request the model to extract keywords based on the normalized query and specified limit
        llm_response = chain.invoke({"query": query, "limit": limit})

        if llm_response is None or not isinstance(llm_response, KeywordList):
            return KeywordList(keywords=[]) # Fail gracefully with an empty keyword list if the model response is invalid
        
        return llm_response

    def gen_keywords_str(self, query: str, limit: int = 5, include_query: bool = True) -> str:
        """
        Generate a BM25-friendly query string from a flat keyword list.

        :param query: The original user query.
        :param limit: Optional maximum number of keywords to extract.
        :param include_query: Whether to prepend the cleaned original query.
        """
        cleaned_query = self.normalize_query(query)
        # Chinese words variants for legal references, e.g. "第18条" -> "第十八条"
        extracted_legal_forms = self.extract_cn_legal_forms(cleaned_query)
        keyword_list = self.gen_keywords(cleaned_query, limit=limit)

        parts: list[str] = []
        # Prepend the cleaned original query if requested
        if include_query:
            parts.append(cleaned_query)

        seen = set()
        for keyword in keyword_list.keywords:
            cleaned_keyword = " ".join(keyword.strip().split())
            if not cleaned_keyword:
                continue
            lowered = cleaned_keyword.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            parts.append(cleaned_keyword)
        
        # Add the Chinese numeral variants of legal references
        for variant in extracted_legal_forms:
            if variant in seen:
                continue
            seen.add(variant)
            parts.append(variant)

        return " ".join(parts)