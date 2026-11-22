
from unittest.mock import patch

from cv_intelligence.agent import build_agent


@patch("cv_intelligence.agent.evaluate_retrieved_candidates")
@patch("cv_intelligence.agent.search_hybrid_candidates")
def test_agent_retrieves_once(mock_search, mock_evaluate):
    mock_search.return_value = [
        {
            "candidate_id": "candidate-a",
            "document_id": "doc-a",
            "best_rrf_score": 0.03,
            "evidence": [],
        }
    ]

    mock_evaluate.return_value = [
        {
            "candidate_id": "candidate-a",
            "document_id": "doc-a",
            "best_rrf_score": 0.03,
            "matched_skills": ["Python"],
            "skills_not_found_in_evidence": [],
            "matched_skill_count": 1,
            "required_skill_count": 1,
            "skill_evidence": {
                "Python": [
                    {
                        "chunk_id": "chunk-a",
                        "section": "skills",
                        "page": 1,
                        "text": "Python developer",
                    }
                ]
            },
        }
    ]

    agent = build_agent()

    result = agent.invoke(
        {
            "query": "Python developer",
            "required_skills": ["Python"],
            "limit": 5,
            "candidates": [],
            "evaluated_candidates": [],
        }
    )

    # Verify hybrid retrieval runs only once.
    mock_search.assert_called_once_with(
        query="Python developer",
        limit=5,
    )

    # Verify evaluation uses the retrieved candidates.
    mock_evaluate.assert_called_once_with(
        candidates=mock_search.return_value,
        required_skills=["Python"],
    )

    # Verify the summary node adds a summary.
    candidate = result["evaluated_candidates"][0]

    assert candidate["candidate_id"] == "candidate-a"
    assert candidate["matched_skills"] == ["Python"]
    assert "summary" in candidate
    assert "Python" in candidate["summary"]
    assert "skills" in candidate["summary"]
