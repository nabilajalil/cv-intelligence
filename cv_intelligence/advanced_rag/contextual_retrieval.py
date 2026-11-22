
from pathlib import Path

from cv_intelligence.advanced_rag.long_context_search import (
    load_cv_documents,
)


def build_document_context(
    filename: str,
    candidate_name: str,
    document_skills: list[str],
) -> str:
    """Build factual document-level context for a CV chunk."""
    skills = ", ".join(document_skills) if document_skills else "Not specified"

    return (
        f"Source document: {filename}\n"
        f"Candidate: {candidate_name}\n"
        f"Skills documented in this CV: {skills}"
    )


def contextualize_chunk(
    chunk: dict,
    document_context: str,
) -> dict:
    """Prepend document context while retaining the original chunk."""
    original_text = chunk["text"]

    contextual_text = (
        f"{document_context}\n"
        f"Section: {chunk.get('section', 'unknown')}\n\n"
        f"Original chunk:\n{original_text}"
    )

    return {
        **chunk,
        "original_text": original_text,
        "contextual_text": contextual_text,
    }


def create_contextual_chunks(
    chunks: list[dict],
    document_context: str,
) -> list[dict]:
    """Contextualize chunks without modifying the originals."""
    return [
        contextualize_chunk(chunk, document_context)
        for chunk in chunks
    ]
