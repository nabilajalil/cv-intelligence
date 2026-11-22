
import argparse
import json
from pathlib import Path

from cv_intelligence.advanced_rag.rag_baseline import (
    retrieve_cv_documents,
)


DATA_PATH = Path("data/public_benchmark/resumes.jsonl")


def load_public_cvs(path: Path = DATA_PATH) -> list[dict]:
    """Load public resumes into the existing BM25 document format."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    documents = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            documents.append({
                "filename": str(record["sample_id"]),
                "text": record["resume_text"],
                "desired_position": record.get("desired_position"),
                "years_experience": record.get("years_experience"),
                "ground_truth": record["ground_truth"],
            })

    return documents


def search_public_cvs(
    query: str,
    documents: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """Run BM25 against public CV text."""
    return retrieve_cv_documents(
        query=query,
        documents=documents,
        top_k=top_k,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        type=str,
        help="Recruiter search query",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    documents = load_public_cvs()

    results = search_public_cvs(
        query=args.query,
        documents=documents,
        top_k=args.top_k,
    )

    print(f"Dataset: {len(documents)} public CVs")
    print(f"Query: {args.query}")
    print(f"Top-k: {args.top_k}\n")

    for rank, candidate in enumerate(results, start=1):
        truth = candidate["ground_truth"]

        name = " ".join(
            part for part in [
                truth.get("first_name"),
                truth.get("last_name"),
            ]
            if part
        )

        skills = [
            item["skill_name"]
            for item in truth.get("skills", [])
            if isinstance(item, dict)
            and isinstance(item.get("skill_name"), str)
        ]

        print(f"=== RANK {rank} ===")
        print("Sample ID:", candidate["filename"])
        print("Candidate:", name or "Unknown")
        print("Position:", candidate["desired_position"])
        print("Years of experience:", candidate["years_experience"])
        print("Documented skills:", ", ".join(skills[:12]))
        print()


if __name__ == "__main__":
    main()
