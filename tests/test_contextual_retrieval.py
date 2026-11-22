
from cv_intelligence.advanced_rag.contextual_retrieval import (
    build_document_context,
    contextualize_chunk,
    create_contextual_chunks,
)


def test_build_document_context():
    context = build_document_context(
        filename="candidate_01.docx",
        candidate_name="Candidate 01",
        document_skills=["Python", "Docker", "AWS"],
    )

    assert "candidate_01.docx" in context
    assert "Candidate 01" in context
    assert "Python, Docker, AWS" in context


def test_contextualize_chunk_preserves_original():
    chunk = {
        "chunk_id": "chunk-1",
        "candidate_id": "candidate-1",
        "section": "projects",
        "text": "Built REST APIs.",
    }

    original_copy = chunk.copy()

    result = contextualize_chunk(
        chunk,
        document_context="Candidate: Candidate 01",
    )

    assert result["original_text"] == "Built REST APIs."
    assert result["text"] == "Built REST APIs."
    assert "Candidate: Candidate 01" in result["contextual_text"]
    assert "Section: projects" in result["contextual_text"]
    assert chunk == original_copy


def test_create_contextual_chunks():
    chunks = [
        {
            "chunk_id": "chunk-1",
            "section": "skills",
            "text": "Python and Docker",
        },
        {
            "chunk_id": "chunk-2",
            "section": "projects",
            "text": "Built backend services",
        },
    ]

    results = create_contextual_chunks(
        chunks,
        document_context="Candidate: Candidate 01",
    )

    assert len(results) == 2
    assert all("contextual_text" in item for item in results)
    assert results[0]["original_text"] == "Python and Docker"
    assert results[1]["original_text"] == "Built backend services"
    assert "Candidate: Candidate 01" in results[1]["contextual_text"]
