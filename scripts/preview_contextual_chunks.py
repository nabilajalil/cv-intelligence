
import json
from pathlib import Path

from openai import OpenAI

from cv_intelligence.config import Settings
from cv_intelligence.store import VectorStore
from cv_intelligence.advanced_rag.contextual_retrieval import (
    build_document_context,
    create_contextual_chunks,
)


def main():
    settings = Settings()
    data_dir = Path(settings.cv_data_dir)

    store = VectorStore(
        path=data_dir / "chroma",
        client=OpenAI(api_key=settings.openai_api_key),
    )

    profile_paths = sorted((data_dir / "profiles").glob("*.json"))

    if not profile_paths:
        raise SystemExit("No candidate profiles found.")

    total_original = 0
    total_contextual = 0
    candidates_processed = 0
    first_example = None

    for profile_path in profile_paths:
        with profile_path.open(encoding="utf-8") as file:
            record = json.load(file)

        candidate_id = record.get("candidate_id")
        profile = record.get("profile", {})

        if not candidate_id or not isinstance(profile, dict):
            continue

        chunks = store.get_candidate_chunks(candidate_id)

        if not chunks:
            continue

        skills = profile.get("skills") or []

        if not isinstance(skills, list):
            skills = []

        skills = [
            str(skill)
            for skill in skills
            if isinstance(skill, str) and skill.strip()
        ]

        context = build_document_context(
            filename=record.get("filename", profile_path.name),
            candidate_name=profile.get("name") or "Unknown candidate",
            document_skills=skills,
        )

        contextual_chunks = create_contextual_chunks(
            chunks=chunks,
            document_context=context,
        )

        candidates_processed += 1
        total_original += len(chunks)
        total_contextual += len(contextual_chunks)

        if first_example is None and contextual_chunks:
            first_example = contextual_chunks[0]

    print("Candidates processed:", candidates_processed)
    print("Original chunks:", total_original)
    print("Contextual chunks:", total_contextual)

    if first_example:
        print("\n=== ORIGINAL CHUNK ===")
        print(first_example["original_text"])

        print("\n=== CONTEXTUALIZED CHUNK ===")
        print(first_example["contextual_text"])


if __name__ == "__main__":
    main()
