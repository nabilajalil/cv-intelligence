
"""Build CV chunks with overlap and summary headers."""

import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.search_public_cvs import load_public_cvs


OUTPUT_PATH = Path(
    "data/public_benchmark/enriched_chunks.jsonl"
)


def build_enriched_chunks(
    chunk_size=800,
    chunk_overlap=150,
):
    """Split CVs and add a small descriptive header."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents = load_public_cvs()
    results = []

    for document in documents:
        candidate_id = document["filename"]
        text = document["text"]

        # Source metadata only: no ground-truth labels.
        position = document.get("desired_position") or "Unknown"

        chunks = splitter.split_text(text)

        for index, chunk in enumerate(chunks):
            header = (
                f"Candidate ID: {candidate_id}\n"
                f"Target position: {position}\n"
                f"Document type: CV\n"
            )

            results.append(
                {
                    "chunk_id": (
                        f"{candidate_id}_enriched_{index:03d}"
                    ),
                    "candidate_id": candidate_id,
                    "chunk_index": index,
                    "text": header + "\n" + chunk,
                    "original_text": chunk,
                }
            )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        for result in results:
            file.write(
                json.dumps(result, ensure_ascii=False)
                + "\n"
            )

    print("=== ENRICHED CV CHUNKS ===")
    print("CVs:", len(documents))
    print("Chunks:", len(results))
    print("Chunk size:", chunk_size)
    print("Chunk overlap:", chunk_overlap)
    print("Saved:", OUTPUT_PATH)

    print("\n=== EXAMPLE CHUNK ===")
    if results:
        print(results[0]["text"][:700])


if __name__ == "__main__":
    build_enriched_chunks()
