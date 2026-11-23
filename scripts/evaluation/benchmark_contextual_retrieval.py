
import json
import re
from pathlib import Path

from openai import OpenAI
from rank_bm25 import BM25Okapi

from cv_intelligence.config import Settings
from cv_intelligence.store import VectorStore
from cv_intelligence.candidate_evaluation import contains_skill
from cv_intelligence.advanced_rag.contextual_retrieval import (
    build_document_context,
    create_contextual_chunks,
)
from cv_intelligence.advanced_rag.long_context_search import (
    load_cv_documents,
)


SKILL_QUERIES = [
    ["Python", "Docker", "AWS"],
    ["Python", "LangGraph"],
    ["RAG", "Docker"],
    ["AWS", "Kubernetes"],
    ["Python", "FastAPI"],
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_indexed_chunks() -> list[dict]:
    settings = Settings()
    data_dir = Path(settings.cv_data_dir)

    store = VectorStore(
        path=data_dir / "chroma",
        client=OpenAI(api_key=settings.openai_api_key),
    )

    all_chunks = []

    for profile_path in sorted((data_dir / "profiles").glob("*.json")):
        with profile_path.open(encoding="utf-8") as file:
            record = json.load(file)

        candidate_id = record.get("candidate_id")
        profile = record.get("profile", {})

        if not candidate_id or not isinstance(profile, dict):
            continue

        chunks = store.get_candidate_chunks(candidate_id)

        skills = profile.get("skills") or []
        if not isinstance(skills, list):
            skills = []

        context = build_document_context(
            filename=record.get("filename", profile_path.name),
            candidate_name=profile.get("name") or "Unknown candidate",
            document_skills=[
                skill for skill in skills
                if isinstance(skill, str) and skill.strip()
            ],
        )

        contextual_chunks = create_contextual_chunks(chunks, context)

        for chunk in contextual_chunks:
            all_chunks.append({
                **chunk,
                "filename": record.get("filename", ""),
            })

    return all_chunks


def retrieve_chunks(
    query: str,
    chunks: list[dict],
    text_field: str,
    top_k: int = 5,
) -> list[dict]:
    corpus = [tokenize(chunk[text_field]) for chunk in chunks]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True,
    )

    return [chunks[index] for index in ranked[:top_k]]


def evaluate_retrieval(
    chunks: list[dict],
    relevant_filenames: set[str],
    skills: list[str],
    text_field: str,
    top_k: int = 5,
) -> dict:
    query = "Find candidates with " + ", ".join(skills)

    retrieved = retrieve_chunks(
        query=query,
        chunks=chunks,
        text_field=text_field,
        top_k=top_k,
    )

    retrieved_filenames = {
        chunk["filename"] for chunk in retrieved
    }

    hits = len(retrieved_filenames & relevant_filenames)

    recall = (
        hits / len(relevant_filenames)
        if relevant_filenames else None
    )

    return {
        "hits": hits,
        "recall": recall,
        "retrieved_filenames": sorted(retrieved_filenames),
    }


def main():
    chunks = load_indexed_chunks()
    documents = load_cv_documents(Path("data/sample_cvs"))

    if not chunks or not documents:
        raise SystemExit("No indexed chunks or CV documents found.")

    print(f"Indexed chunks: {len(chunks)}")
    print(f"Original CVs: {len(documents)}")
    print("Top-k chunks: 5\n")

    for skills in SKILL_QUERIES:
        relevant = {
            doc["filename"]
            for doc in documents
            if all(
                contains_skill(doc["text"], skill)
                for skill in skills
            )
        }

        original = evaluate_retrieval(
            chunks, relevant, skills, "original_text"
        )

        contextual = evaluate_retrieval(
            chunks, relevant, skills, "contextual_text"
        )

        print("Skills:", ", ".join(skills))
        print("Relevant CVs:", len(relevant))
        print(
            "Original BM25 Recall@5:",
            "N/A" if original["recall"] is None
            else f"{original['recall']:.2f}",
        )
        print(
            "Contextual BM25 Recall@5:",
            "N/A" if contextual["recall"] is None
            else f"{contextual['recall']:.2f}",
        )
        print("Original retrieved:", original["retrieved_filenames"])
        print("Contextual retrieved:", contextual["retrieved_filenames"])
        print("-" * 50)


if __name__ == "__main__":
    main()
