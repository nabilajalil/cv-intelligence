
import argparse
import json
from pathlib import Path

from openai import OpenAI

from cv_intelligence.config import Settings
from scripts.evaluate_chunk_bm25 import load_chunks
from scripts.search_public_cvs import load_public_cvs


OUTPUT_PATH = Path(
    "data/public_benchmark/llm_contextual_chunks.jsonl"
)


def generate_context(
    client: OpenAI,
    model: str,
    full_cv: str,
    chunk_text: str,
) -> str:
    """Generate a short, evidence-grounded context for one chunk."""

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You generate retrieval context for CV chunks. "
                    "Explain what the chunk represents within the "
                    "candidate's full CV. Use only information explicitly "
                    "supported by the supplied CV. Do not invent skills, "
                    "experience, employers, or qualifications. "
                    "Do not copy a generic list of all CV skills. "
                    "Write one or two concise sentences. "
                    "Do not include personal contact information."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"FULL CV:\n{full_cv}\n\n"
                    f"CHUNK:\n{chunk_text}\n\n"
                    "Write context specifically for this chunk."
                ),
            },
        ],
    )

    return (response.choices[0].message.content or "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=50,
        help="Maximum number of chunks to process",
    )
    args = parser.parse_args()

    if args.limit < 1 or args.max_chunks < 1:
        raise ValueError("Limits must be positive.")

    settings = Settings()
    client = OpenAI(api_key=settings.openai_api_key)

    documents = load_public_cvs()[:args.limit]
    selected_ids = {
        document["filename"]
        for document in documents
    }

    full_text_by_id = {
        document["filename"]: document["text"]
        for document in documents
    }

    chunks = [
        chunk
        for chunk in load_chunks()
        if chunk["candidate_id"] in selected_ids
    ]

    if len(chunks) > args.max_chunks:
        raise ValueError(
            f"Selected {len(chunks)} chunks, exceeding "
            f"--max-chunks={args.max_chunks}. "
            "Increase the cap explicitly to continue."
        )

    print("=== LLM CONTEXT GENERATION ===")
    print("CVs:", len(documents))
    print("Chunks:", len(chunks))
    print("Model:", settings.openai_extraction_model)
    print("API calls required:", len(chunks))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Resume safely if a previous run stopped partway through.
    completed = {}

    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    record = json.loads(line)
                    completed[record["chunk_id"]] = record

    with OUTPUT_PATH.open("a", encoding="utf-8") as file:
        for index, chunk in enumerate(chunks, start=1):
            chunk_id = chunk["chunk_id"]

            if chunk_id in completed:
                print(f"[{index}/{len(chunks)}] Skipping {chunk_id}")
                continue

            context = generate_context(
                client=client,
                model=settings.openai_extraction_model,
                full_cv=full_text_by_id[chunk["candidate_id"]],
                chunk_text=chunk["text"],
            )

            contextual_chunk = {
                **chunk,
                "original_text": chunk["text"],
                "context": context,
                "text": f"{context}\n\n{chunk['text']}",
            }

            file.write(
                json.dumps(contextual_chunk, ensure_ascii=False)
                + "\n"
            )
            file.flush()

            print(f"[{index}/{len(chunks)}] {chunk_id}")
            print("Context:", context[:180])

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
