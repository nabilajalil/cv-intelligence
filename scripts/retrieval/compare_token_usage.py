
import argparse
import time
from pathlib import Path

from cv_intelligence.advanced_rag.long_context_search import (
    load_cv_documents,
    search_with_long_context_metrics,
)

from cv_intelligence.advanced_rag.rag_baseline import (
    retrieve_cv_documents,
)

from cv_intelligence.advanced_rag.cost_calculator import (
    estimate_api_cost,
)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("query")

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

    if args.limit_documents < 1 or args.top_k < 1:
        parser.error("Document limit and top-k must be positive.")

    documents = load_cv_documents(
        Path("data/sample_cvs")
    )[:args.limit_documents]

    if not documents:
        raise SystemExit("No CV documents found.")

    print(f"Dataset: {len(documents)} CVs")

    print("\n=== LONG CONTEXT ===")

    start = time.perf_counter()

    long_result, long_metrics = search_with_long_context_metrics(
        query=args.query,
        documents=documents,
    )

    long_latency = time.perf_counter() - start

    print("Matches:", [c.filename for c in long_result.candidates])
    print("Metrics:", long_metrics)
    print(f"Latency: {long_latency:.2f}s")

    print("\n=== RAG ===")

    start = time.perf_counter()

    retrieved = retrieve_cv_documents(
        query=args.query,
        documents=documents,
        top_k=args.top_k,
    )

    rag_result, rag_metrics = search_with_long_context_metrics(
        query=args.query,
        documents=retrieved,
    )

    rag_latency = time.perf_counter() - start

    print("Retrieved:", [d["filename"] for d in retrieved])
    print("Matches:", [c.filename for c in rag_result.candidates])
    print("Metrics:", rag_metrics)
    print(f"Latency: {rag_latency:.2f}s")

    print("\n=== COMPARISON ===")

    print(
        f"{'Metric':<24}"
        f"{'Long Context':>16}"
        f"{'RAG':>16}"
    )

    print("-" * 56)

    for label, key in [
        ("Input tokens", "input_tokens"),
        ("Output tokens", "output_tokens"),
        ("Total tokens", "total_tokens"),
        ("Documents sent", "documents_sent"),
        ("Context characters", "context_characters"),
    ]:
        print(
            f"{label:<24}"
            f"{str(long_metrics[key]):>16}"
            f"{str(rag_metrics[key]):>16}"
        )

    print(
        f"{'Latency (seconds)':<24}"
        f"{long_latency:>16.2f}"
        f"{rag_latency:>16.2f}"
    )

    long_tokens = long_metrics["input_tokens"]
    rag_tokens = rag_metrics["input_tokens"]

    if long_tokens and rag_tokens is not None:
        reduction = (long_tokens - rag_tokens) / long_tokens * 100

        print(
            f"\nRAG input-token reduction: {reduction:.1f}%"
        )

        print("\n=== ESTIMATED API COST ===")

    # Example rates in USD per million tokens.
    # Verify current rates before using these for billing estimates.
    input_rate = 0.40
    output_rate = 1.60

    long_cost = estimate_api_cost(
        input_tokens=long_metrics["input_tokens"],
        output_tokens=long_metrics["output_tokens"],
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
    )

    rag_cost = estimate_api_cost(
        input_tokens=rag_metrics["input_tokens"],
        output_tokens=rag_metrics["output_tokens"],
        input_price_per_million=input_rate,
        output_price_per_million=output_rate,
    )

    print(f"Model: {long_metrics['model']}")
    print(f"Input rate:  ${input_rate:.2f} / 1M tokens")
    print(f"Output rate: ${output_rate:.2f} / 1M tokens")
    print()
    print(f"Long Context: ${long_cost:.6f}")
    print(f"RAG:          ${rag_cost:.6f}")

    if long_cost > 0:
        savings = (long_cost - rag_cost) / long_cost * 100
        print(f"Estimated cost reduction: {savings:.1f}%")

    print(
        "\nNote: Estimated inference cost only. "
        "Excludes indexing, embedding, and other infrastructure costs."
    )


if __name__ == "__main__":
    main()

