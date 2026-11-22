
import json
import re
from pathlib import Path


DATA_DIR = Path("data/public_benchmark")
INPUT_PATH = DATA_DIR / "resumes.jsonl"
OUTPUT_PATH = DATA_DIR / "chunks.jsonl"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split resume text into overlapping character-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def build_chunks(
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> list[dict]:
    """Create chunks with candidate and source metadata."""

    chunks = []

    with input_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            candidate_id = record["sample_id"]
            resume_text = record["resume_text"]

            text_chunks = split_text(resume_text)

            for index, text in enumerate(text_chunks):
                chunks.append({
                    "chunk_id": f"{candidate_id}_chunk_{index:03d}",
                    "candidate_id": candidate_id,
                    "chunk_index": index,
                    "text": text,
                })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    return chunks


def main():
    chunks = build_chunks()

    candidate_ids = {
        chunk["candidate_id"]
        for chunk in chunks
    }

    print("=== PUBLIC CV CHUNKING ===")
    print("Candidates:", len(candidate_ids))
    print("Total chunks:", len(chunks))
    print("Chunk size:", CHUNK_SIZE)
    print("Chunk overlap:", CHUNK_OVERLAP)
    print("Output:", OUTPUT_PATH)

    if chunks:
        print("\n=== FIRST CHUNK ===")
        print("Chunk ID:", chunks[0]["chunk_id"])
        print("Candidate ID:", chunks[0]["candidate_id"])
        print("Chunk index:", chunks[0]["chunk_index"])
        print("Text length:", len(chunks[0]["text"]))
        print("\nText preview:")
        print(chunks[0]["text"][:500])


if __name__ == "__main__":
    main()
