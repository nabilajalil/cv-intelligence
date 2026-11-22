
import argparse
import json
import time
from pathlib import Path

from cv_intelligence.advanced_rag.long_context_search import (
    load_cv_documents,
    search_with_long_context,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        help="Recruiter search request",
    )
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=5,
        help="Number of CVs to include in the experiment",
    )

    args = parser.parse_args()

    directory = Path("data/sample_cvs")
    documents = load_cv_documents(directory)

    if not documents:
        raise SystemExit(
            "No DOCX CVs found in data/sample_cvs."
        )

    documents = documents[: args.limit_documents]

    print(f"Loaded {len(documents)} CV documents")
    print(
        "Context characters:",
        sum(len(doc["text"]) for doc in documents),
    )

    start = time.perf_counter()

    result = search_with_long_context(
        query=args.query,
        documents=documents,
    )

    elapsed = time.perf_counter() - start

    print()
    print(result.model_dump_json(indent=2))
    print()
    print(f"LLM elapsed time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
