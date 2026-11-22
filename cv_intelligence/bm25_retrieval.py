
import re

import chromadb
from rank_bm25 import BM25Okapi

from .config import Settings


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase searchable tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def search_bm25(query: str, limit: int = 5) -> list[dict]:
    """Search stored CV chunks using BM25 keyword ranking."""

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    if limit < 1:
        raise ValueError("Limit must be greater than zero")

    settings = Settings()

    client = chromadb.PersistentClient(
        path=str(settings.cv_data_dir / "chroma")
    )
    collection = client.get_collection("cv_chunks")

    data = collection.get(include=["documents", "metadatas"])

    documents = data["documents"]
    metadatas = data["metadatas"]
    chunk_ids = data["ids"]

    if not documents:
        return []

    tokenized_documents = [
        tokenize(document) for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(documents)),
        key=lambda i: scores[i],
        reverse=True,
    )

    results = []

    for index in ranked_indices[:limit]:
        if scores[index] <= 0:
            continue

        metadata = metadatas[index]

        results.append(
            {
                "chunk_id": chunk_ids[index],
                "candidate_id": metadata["candidate_id"],
                "document_id": metadata["document_id"],
                "section": metadata["section"],
                "page": metadata["page"],
                "text": documents[index],
                "bm25_score": float(scores[index]),
            }
        )

    return results
