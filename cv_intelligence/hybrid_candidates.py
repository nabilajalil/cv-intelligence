
from .hybrid_retrieval import search_hybrid


def search_hybrid_candidates(
    query: str,
    limit: int = 5,
    chunk_pool: int = 100,
) -> list[dict]:
    """Rank distinct candidates using hybrid retrieval."""

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    if limit < 1:
        raise ValueError("limit must be positive")

    if chunk_pool < 1:
        raise ValueError("chunk_pool must be positive")

    chunks = search_hybrid(
        query=query,
        limit=max(limit, chunk_pool),
        pool_size=max(limit, chunk_pool),
    )

    candidates = {}

    for chunk in chunks:
        candidate_id = chunk["candidate_id"]

        if candidate_id not in candidates:
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "document_id": chunk["document_id"],
                "best_rrf_score": chunk["rrf_score"],
                "evidence": [],
            }

        candidate = candidates[candidate_id]

        candidate["best_rrf_score"] = max(
            candidate["best_rrf_score"],
            chunk["rrf_score"],
        )

        candidate["evidence"].append(
            {
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "page": chunk["page"],
                "text": chunk["text"],
                "rrf_score": chunk["rrf_score"],
                "retrieved_by": chunk["retrieved_by"],
            }
        )

    ranked = sorted(
        candidates.values(),
        key=lambda item: (
            -item["best_rrf_score"],
            item["candidate_id"],
        ),
    )

    for candidate in ranked:
        candidate["evidence"] = sorted(
            candidate["evidence"],
            key=lambda item: -item["rrf_score"],
        )[:3]

    return ranked[:limit]
