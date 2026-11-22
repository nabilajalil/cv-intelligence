
from cv_intelligence.advanced_rag.long_context_search import (
    build_long_context,
)


def test_build_long_context():
    documents = [
        {
            "filename": "candidate_a.docx",
            "text": "Python, Docker",
        },
        {
            "filename": "candidate_b.docx",
            "text": "AWS, Kubernetes",
        },
    ]

    context = build_long_context(documents)

    assert "candidate_a.docx" in context
    assert "candidate_b.docx" in context
    assert "Python, Docker" in context
    assert "AWS, Kubernetes" in context
    assert context.count("=== END CV ===") == 2
