
import re

from rank_bm25 import BM25Okapi

from cv_intelligence.advanced_rag.long_context_search import (
    search_with_long_context,
)


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 retrieval."""
    return re.findall(r"\w+", text.lower())


def retrieve_cv_documents(
    query: str,
    documents: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Retrieve the most relevant CVs using BM25."""
    if not documents:
        return []

    tokenized_documents = [
        tokenize(document["text"])
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda index: scores[index],
        reverse=True,
    )

    return [
        documents[index]
        for index in ranked_indices[:top_k]
    ]


def search_with_rag(
    query: str,
    documents: list[dict],
    top_k: int = 3,
):
    """Retrieve CVs, then ask the LLM to analyze them."""
    retrieved_documents = retrieve_cv_documents(
        query=query,
        documents=documents,
        top_k=top_k,
    )

    if not retrieved_documents:
        raise ValueError("No CV documents available for retrieval.")

    return search_with_long_context(
        query=query,
        documents=retrieved_documents,
    )
