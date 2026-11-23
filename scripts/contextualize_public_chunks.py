
import json
from collections import Counter
from pathlib import Path

from scripts.evaluate_chunk_bm25 import load_chunks
from scripts.retrieval.search_public_cvs import load_public_cvs


OUTPUT_PATH = Path(
    "data/public_benchmark/contextual_chunks.jsonl"
)


def extract_role_from_text(resume_text: str) -> str:
    """Extract the role from the resume header, not ground truth."""
    lines = [
        line.strip()
        for line in resume_text.splitlines()
        if line.strip()
    ]

    # Dataset resumes typically begin with:
    # line 1: candidate name
    # line 2: job title
    if len(lines) >= 2:
        return lines[1]

    return "Unknown role"



import re


TECHNOLOGIES = [
    "Python",
    "Docker",
    "AWS",
    "Java",
    "Linux",
    "JavaScript",
    "GitLab",
    "SQL",
    "Kubernetes",
    "Azure",
]


def extract_technologies(resume_text: str) -> list[str]:
    """Find explicitly mentioned technologies in original CV text."""

    found = []

    for technology in TECHNOLOGIES:
        pattern = (
            r"(?<!\w)"
            + re.escape(technology)
            + r"(?!\w)"
        )

        if re.search(pattern, resume_text, flags=re.IGNORECASE):
            found.append(technology)

    return found


def contextualize_chunks(
    chunks: list[dict],
    documents: list[dict],
) -> list[dict]:
    """Add source-document context without using evaluation labels."""

    roles = {
        document["filename"]: extract_role_from_text(
            document["text"]
        )
        for document in documents
    }



    technologies_by_candidate = {
        document["filename"]: extract_technologies(
            document["text"]
        )
        for document in documents
    }

    
    counts = Counter(
        chunk["candidate_id"]
        for chunk in chunks
    )

    contextual_chunks = []

    for chunk in chunks:
        candidate_id = chunk["candidate_id"]
        role = roles.get(candidate_id, "Unknown role")

        technologies = technologies_by_candidate.get(
            candidate_id, []
        )

        technology_context = (
            ", ".join(technologies)
            if technologies
            else "None explicitly identified"
        )

        context = (
            f"Source CV role: {role}. "
            f"Technologies mentioned in source CV: "
            f"{technology_context}. "
            f"Chunk {chunk['chunk_index'] + 1} "
            f"of {counts[candidate_id]}."
        )

        contextual_chunks.append({
            **chunk,
            "original_text": chunk["text"],
            "context": context,
            "text": f"{context}\n\n{chunk['text']}",
        })

    return contextual_chunks


def main():
    chunks = load_chunks()
    documents = load_public_cvs()

    contextual_chunks = contextualize_chunks(
        chunks,
        documents,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for chunk in contextual_chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    print("=== CONTEXTUAL CHUNK GENERATION ===")
    print("Original chunks:", len(chunks))
    print("Contextual chunks:", len(contextual_chunks))
    print("Output:", OUTPUT_PATH)

    if contextual_chunks:
        example = contextual_chunks[0]

        print("\n=== FIRST CONTEXTUAL CHUNK ===")
        print("Chunk ID:", example["chunk_id"])
        print("Context:", example["context"])
        print("\nContextual text preview:")
        print(example["text"][:400])


if __name__ == "__main__":
    main()
