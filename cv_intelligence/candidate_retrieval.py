
from .retrieval import search_candidates


def search_unique_candidates(
    query: str,
    limit: int = 5,
    chunk_pool: int = 50,
) -> list[dict]:
    """Return distinct candidates with supporting evidence."""

    if limit < 1:
        raise ValueError("Limit must be greater than zero")

    if chunk_pool < 1:
        raise ValueError("chunk_pool must be greater than zero")

    # Fetch more chunks than the number of candidates needed.
    chunks = search_candidates(
        query=query,
        limit=max(limit, chunk_pool),
    )

    candidates = {}

    for chunk in chunks:
        candidate_id = chunk["candidate_id"]

        if candidate_id not in candidates:
            candidates[candidate_id] = {
                "candidate_id": candidate_id,
                "document_id": chunk["document_id"],
                "best_distance": chunk["distance"],
                "evidence": [],
            }

        candidates[candidate_id]["evidence"].append(
            {
                "section": chunk["section"],
                "page": chunk["page"],
                "text": chunk["text"],
                "distance": chunk["distance"],
            }
        )

        candidates[candidate_id]["best_distance"] = min(
            candidates[candidate_id]["best_distance"],
            chunk["distance"],
        )

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: candidate["best_distance"],
    )

    # Keep the 3 strongest chunks per candidate.
    for candidate in ranked:
        candidate["evidence"] = sorted(
            candidate["evidence"],
            key=lambda item: item["distance"],
        )[:3]

    return ranked[:limit]
