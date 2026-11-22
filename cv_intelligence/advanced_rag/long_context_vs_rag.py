
from pathlib import Path

from cv_intelligence.config import Settings
from cv_intelligence.store import VectorStore


def load_full_candidate_context(
    candidate_id: str,
    store: VectorStore | None = None,
) -> str:
    """
    Reconstruct a candidate's indexed CV text from ChromaDB.

    This uses all indexed chunks, not just the top search results.
    """
    if store is None:
        from openai import OpenAI

        settings = Settings()

        store = VectorStore(
            path=Path(settings.cv_data_dir) / "chroma",
            client=OpenAI(api_key=settings.openai_api_key),
        )

    chunks = store.get_candidate_chunks(candidate_id)

    if not chunks:
        return ""

    chunks = sorted(
        chunks,
        key=lambda chunk: (
            str(chunk.get("section", "")),
            int(chunk.get("page") or 0),
            str(chunk.get("chunk_id", "")),
        ),
    )

    parts = []

    for chunk in chunks:
        section = chunk.get("section", "unknown")
        page = chunk.get("page", "unknown")
        text = chunk.get("text", "")

        parts.append(
            f"[Section: {section} | Page: {page}]\n{text}"
        )

    return "\n\n".join(parts)


def compare_context_sizes(
    candidate_id: str,
    retrieved_evidence: list[dict],
    store: VectorStore | None = None,
) -> dict:
    """
    Compare full indexed CV context against retrieved snippets.

    Character counts are used as an initial size comparison.
    They are NOT token counts.
    """
    full_context = load_full_candidate_context(
        candidate_id,
        store=store,
    )

    retrieved_context = "\n\n".join(
        chunk.get("text", "")
        for chunk in retrieved_evidence
    )

    return {
        "candidate_id": candidate_id,
        "full_context_chars": len(full_context),
        "retrieved_context_chars": len(retrieved_context),
        "full_context": full_context,
        "retrieved_context": retrieved_context,
    }
