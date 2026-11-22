
import argparse
import time
from pathlib import Path

from cv_intelligence.advanced_rag.long_context_search import (
    load_cv_documents,
    search_with_long_context,
)
from cv_intelligence.advanced_rag.rag_baseline import (
    retrieve_cv_documents,
    search_with_rag,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "query",
        help="Recruiter search query",
    )
    parser.add_argument(
        "--limit-documents",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    documents = load_cv_documents(Path("data/sample_cvs"))
    documents = documents[:args.limit_documents]

    if not documents:
        raise SystemExit("No CV documents found.")

    print(f"Dataset: {len(documents)} CVs")
    print(f"RAG top-k: {args.top_k}")

    print("\n=== LONG CONTEXT ===")
    start = time.perf_counter()

    long_result = search_with_long_context(
        query=args.query,
        documents=documents,
    )

    long_latency = time.perf_counter() - start

    print(long_result.model_dump_json(indent=2))
    print(f"Latency: {long_latency:.2f}s")

    print("\n=== RAG ===")

    retrieved = retrieve_cv_documents(
        query=args.query,
        documents=documents,
        top_k=args.top_k,
    )

    print(
        "Retrieved CVs:",
        [doc["filename"] for doc in retrieved],
    )

    start = time.perf_counter()

    rag_result = search_with_rag(
        query=args.query,
        documents=documents,
        top_k=args.top_k,
    )

    rag_latency = time.perf_counter() - start

    print(rag_result.model_dump_json(indent=2))
    print(f"LLM + retrieval latency: {rag_latency:.2f}s")

    print("\n=== COMPARISON ===")
    print(f"Long Context: {long_latency:.2f}s")
    print(f"RAG:          {rag_latency:.2f}s")

    print(
        "Long Context matches:",
        [c.filename for c in long_result.candidates],
    )
    print(
        "RAG matches:",
        [c.filename for c in rag_result.candidates],
    )


if __name__ == "__main__":
    main()
