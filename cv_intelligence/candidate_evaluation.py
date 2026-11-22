
import re
from openai import OpenAI

from .config import Settings
from .store import VectorStore
from .hybrid_candidates import search_hybrid_candidates


def contains_skill(text: str, skill: str) -> bool:
    """Check for a whole-term skill mention, case-insensitively."""
    pattern = (
        r"(?<!\w)"
        + re.escape(skill)
        + r"(?!\w)"
    )
    return bool(re.search(pattern, text, re.IGNORECASE))



def evaluate_retrieved_candidates(
    candidates: list[dict],
    required_skills: list[str],
) -> list[dict]:
    """Evaluate already-retrieved candidates against indexed CV text."""

    if not required_skills:
        raise ValueError("At least one required skill is needed")

    if any(not skill.strip() for skill in required_skills):
        raise ValueError("Required skills cannot be blank")

    settings = Settings()

    store = VectorStore(
        path=settings.cv_data_dir / "chroma",
        model=settings.openai_embedding_model,
        client=OpenAI(api_key=settings.openai_api_key),
    )

    evaluated = []

    for candidate in candidates:
        all_chunks = store.get_candidate_chunks(
            candidate["candidate_id"]
        )

        matched = []
        not_found = []
        skill_evidence = {}

        for skill in required_skills:
            supporting_chunks = [
                chunk
                for chunk in all_chunks
                if contains_skill(chunk["text"], skill)
            ]

            if supporting_chunks:
                matched.append(skill)
                skill_evidence[skill] = [
                    {
                        "chunk_id": chunk["chunk_id"],
                        "section": chunk["section"],
                        "page": chunk["page"],
                        "text": chunk["text"],
                    }
                    for chunk in supporting_chunks
                ]
            else:
                not_found.append(skill)

        evaluated.append(
            {
                "candidate_id": candidate["candidate_id"],
                "document_id": candidate["document_id"],
                "best_rrf_score": candidate["best_rrf_score"],
                "matched_skills": matched,
                "skills_not_found_in_evidence": not_found,
                "matched_skill_count": len(matched),
                "required_skill_count": len(required_skills),
                "skill_evidence": skill_evidence,
            }
        )

    evaluated.sort(
        key=lambda item: (
            -item["matched_skill_count"],
            -item["best_rrf_score"],
            item["candidate_id"],
        )
    )

    return evaluated


def evaluate_candidates(
    query: str,
    required_skills: list[str],
    limit: int = 5,
) -> list[dict]:
    """Retrieve candidates, then evaluate their indexed evidence."""

    if not required_skills:
        raise ValueError("At least one required skill is needed")

    if any(not skill.strip() for skill in required_skills):
        raise ValueError("Required skills cannot be blank")

    candidates = search_hybrid_candidates(
        query=query,
        limit=limit,
    )

    return evaluate_retrieved_candidates(
        candidates=candidates,
        required_skills=required_skills,
    )
