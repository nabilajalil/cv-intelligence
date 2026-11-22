
from unittest.mock import patch

from cv_intelligence.candidate_evaluation import (
    contains_skill,
    evaluate_candidates,
)


def test_contains_skill():
    assert contains_skill("Python, Docker, AWS", "Docker")
    assert contains_skill("python developer", "Python")
    assert not contains_skill("Python developer", "Java")



@patch("cv_intelligence.candidate_evaluation.VectorStore")
@patch("cv_intelligence.candidate_evaluation.OpenAI")
@patch(
    "cv_intelligence.candidate_evaluation.search_hybrid_candidates"
)
def test_skill_evaluation(mock_search, mock_openai, mock_store):
    mock_search.return_value = [
        {
            "candidate_id": "candidate-a",
            "document_id": "doc-a",
            "best_rrf_score": 0.03,
            "evidence": [],
        }
    ]

    mock_store.return_value.get_candidate_chunks.return_value = [
        {
            "chunk_id": "chunk-a",
            "candidate_id": "candidate-a",
            "document_id": "doc-a",
            "section": "skills",
            "page": 1,
            "text": "Python, Docker",
        }
    ]

    results = evaluate_candidates(
        "Python Docker AWS",
        required_skills=["Python", "Docker", "AWS"],
    )

    assert results[0]["matched_skills"] == [
        "Python",
        "Docker",
    ]
    assert results[0]["skills_not_found_in_evidence"] == [
        "AWS"
    ]
    assert results[0]["matched_skill_count"] == 2