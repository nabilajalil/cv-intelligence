
from unittest.mock import patch

from cv_intelligence.hybrid_retrieval import search_hybrid


@patch("cv_intelligence.hybrid_retrieval.search_bm25")
@patch("cv_intelligence.hybrid_retrieval.search_candidates")
def test_rrf_combines_rankings(mock_dense, mock_bm25):
    def chunk(chunk_id):
        return {
            "chunk_id": chunk_id,
            "candidate_id": f"candidate-{chunk_id}",
            "document_id": f"document-{chunk_id}",
            "section": "skills",
            "page": 1,
            "text": "Python Docker AWS",
        }

    mock_dense.return_value = [
        chunk("a"),
        chunk("b"),
    ]

    mock_bm25.return_value = [
        chunk("b"),
        chunk("c"),
    ]

    results = search_hybrid("Python Docker AWS")

    assert len(results) == 3

    # Chunk b appears in both retrieval lists.
    assert results[0]["chunk_id"] == "b"
    assert set(results[0]["retrieved_by"]) == {
        "semantic",
        "bm25",
    }

    expected_score = 1 / 62 + 1 / 61
    assert abs(
        results[0]["rrf_score"] - expected_score
    ) < 1e-9
