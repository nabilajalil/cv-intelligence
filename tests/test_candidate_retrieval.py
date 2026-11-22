
from unittest.mock import patch

from cv_intelligence.candidate_retrieval import (
    search_unique_candidates,
)


@patch("cv_intelligence.candidate_retrieval.search_candidates")
def test_unique_candidate_results(mock_search):
    mock_search.return_value = [
        {
            "candidate_id": "candidate-a",
            "document_id": "doc-a",
            "section": "skills",
            "page": 1,
            "text": "Python, Docker",
            "distance": 0.3,
        },
        {
            "candidate_id": "candidate-a",
            "document_id": "doc-a",
            "section": "projects",
            "page": 1,
            "text": "AWS project",
            "distance": 0.5,
        },
        {
            "candidate_id": "candidate-b",
            "document_id": "doc-b",
            "section": "skills",
            "page": 1,
            "text": "Python, AWS",
            "distance": 0.4,
        },
    ]

    results = search_unique_candidates(
        "Python Docker AWS",
        limit=2,
    )

    assert len(results) == 2
    assert results[0]["candidate_id"] == "candidate-a"
    assert results[1]["candidate_id"] == "candidate-b"
    assert len(results[0]["evidence"]) == 2
    assert results[0]["best_distance"] == 0.3
