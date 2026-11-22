
from .retrieval import search_candidates
from .bm25_retrieval import search_bm25


def search_hybrid(
    query: str,
    limit: int = 5,
    pool_size: int = 50,
    rrf_k: int = 60,
) -> list[dict]:
    """Combine semantic and BM25 chunk rankings using RRF."""

    if not query.strip():
        raise ValueError("Search query cannot be empty")

    if limit < 1 or pool_size < 1:
        raise ValueError("limit and pool_size must be positive")

    if rrf_k < 1:
        raise ValueError("rrf_k must be positive")

    retrieval_limit = max(limit, pool_size)

    dense_results = search_candidates(
        query, limit=retrieval_limit
    )
    sparse_results = search_bm25(
        query, limit=retrieval_limit
    )

    fused = {}

    for source, results in (
        ("semantic", dense_results),
        ("bm25", sparse_results),
    ):
        for rank, chunk in enumerate(results, start=1):
            chunk_id = chunk["chunk_id"]

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "candidate_id": chunk["candidate_id"],
                    "document_id": chunk["document_id"],
                    "section": chunk["section"],
                    "page": chunk["page"],
                    "text": chunk["text"],
                    "rrf_score": 0.0,
                    "retrieved_by": [],
                }

            fused[chunk_id]["rrf_score"] += (
                1.0 / (rrf_k + rank)
            )
            fused[chunk_id]["retrieved_by"].append(source)

    ranked = sorted(
        fused.values(),
        key=lambda item: (
            -item["rrf_score"],
            item["chunk_id"],
        ),
    )

    return ranked[:limit]
