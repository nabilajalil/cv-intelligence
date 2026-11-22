
from pathlib import Path

from openai import OpenAI

from .config import Settings
from .store import VectorStore


def search_candidates(query: str, limit: int = 5) -> list[dict]:
    """Search indexed CV chunks using OpenAI embeddings."""

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    if limit < 1:
        raise ValueError("Limit must be greater than zero")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    store = VectorStore(
        path=settings.cv_data_dir / "chroma",
        model=settings.openai_embedding_model,
        client=client,
    )

    # Avoid calling OpenAI if the database is empty.
    count = store.collection.count()
    if count == 0:
        return []

    raw = store.search(query, limit=min(limit, count))

    results = []

    ids = raw["ids"][0]
    documents = raw["documents"][0]
    metadatas = raw["metadatas"][0]
    distances = raw["distances"][0]

    for chunk_id, text, metadata, distance in zip(
        ids, documents, metadatas, distances
    ):
        results.append(
            {
                "chunk_id": chunk_id,
                "candidate_id": metadata["candidate_id"],
                "document_id": metadata["document_id"],
                "section": metadata["section"],
                "page": metadata["page"],
                "text": text,
                "distance": float(distance),
            }
        )

    return results
