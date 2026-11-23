
import json

from cv_intelligence.advanced_rag.rag_baseline import (
    retrieve_cv_documents,
)
from scripts.retrieval.search_public_cvs import load_public_cvs


def test_load_public_cvs(tmp_path):
    path = tmp_path / "resumes.jsonl"

    sample = {
        "sample_id": "r00001",
        "resume_text": "Python Docker AWS",
        "desired_position": "Backend Developer",
        "years_experience": 5,
        "ground_truth": {
            "first_name": "Test",
            "last_name": "Candidate",
            "skills": [
                {"skill_name": "Python", "level": 3},
            ],
        },
    }

    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    documents = load_public_cvs(path)

    assert len(documents) == 1
    assert documents[0]["filename"] == "r00001"
    assert documents[0]["text"] == "Python Docker AWS"


def test_bm25_retrieves_relevant_document():
    documents = [
        {
            "filename": "candidate_python",
            "text": "Python Docker AWS cloud backend services",
        },
        {
            "filename": "candidate_design",
            "text": "Figma typography user interface design",
        },
        {
            "filename": "candidate_support",
            "text": "Customer support helpdesk ticket management",
        },
    ]

    results = retrieve_cv_documents(
        query="Python Docker AWS",
        documents=documents,
        top_k=1,
    )

    assert results[0]["filename"] == "candidate_python"
