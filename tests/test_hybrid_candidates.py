
from unittest.mock import patch

from cv_intelligence.hybrid_candidates import (
    search_hybrid_candidates,
)


@patch("cv_intelligence.hybrid_candidates.search_hybrid")
def test_hybrid_candidates_are_unique(mock_search):
    def chunk(chunk_id, candidate_id, score):
        return {
            "chunk_id": chunk_id,
            "candidate_id": candidate_id,
            "document_id": f"doc-{candidate_id}",
            "section": "skills",
            "page": 1,
            "text": "Python Docker AWS",
            "rrf_score": score,
            "retrieved_by": ["semantic", "bm25"],
        }

    mock_search.return_value = [
        chunk("a1", "candidate-a", 0.032),
        chunk("a2", "candidate-a", 0.030),
        chunk("b1", "candidate-b", 0.031),
    ]

    results = search_hybrid_candidates(
        "Python Docker AWS",
        limit=2,
    )

    assert len(results) == 2
    assert results[0]["candidate_id"] == "candidate-a"
    assert results[1]["candidate_id"] == "candidate-b"
    assert len(results[0]["evidence"]) == 2
    assert results[0]["best_rrf_score"] == 0.032
