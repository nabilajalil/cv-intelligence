
from scripts.evaluate_chunk_bm25 import (
    load_chunks,
    retrieve_candidates,
)
from scripts.evaluate_public_bm25 import (
    load_queries,
    load_qrels,
    ndcg_at_k,
)


CONTEXTUAL_PATH = (
    "data/public_benchmark/contextual_chunks.jsonl"
)

TOP_K = 5


def evaluate(chunks, queries, qrels):
    results = []

    for query in queries:
        query_id = query["query_id"]

        candidate_ids = retrieve_candidates(
            query=query["text"],
            chunks=chunks,
            top_k=TOP_K,
        )

        labels = qrels[query_id]

        grades = [
            labels.get(candidate_id, 0)
            for candidate_id in candidate_ids
        ]

        total_relevant = sum(
            grade == 2
            for grade in labels.values()
        )

        retrieved_relevant = sum(
            grade == 2
            for grade in grades
        )

        precision = retrieved_relevant / TOP_K

        recall = (
            retrieved_relevant / total_relevant
            if total_relevant else None
        )

        ndcg = ndcg_at_k(
            grades,
            list(labels.values()),
            TOP_K,
        )

        results.append({
            "query_id": query_id,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "candidate_ids": candidate_ids,
            "grades": grades,
        })

    return results


def mean_metric(results, metric):
    values = [
        result[metric]
        for result in results
        if result[metric] is not None
    ]

    return sum(values) / len(values) if values else None


def main():
    from pathlib import Path

    original_chunks = load_chunks()

    contextual_chunks = load_chunks(
        Path(CONTEXTUAL_PATH)
    )

    queries = load_queries()
    qrels = load_qrels()

    print("=== CONTEXTUAL RETRIEVAL COMPARISON ===")
    print("Original chunks:", len(original_chunks))
    print("Contextual chunks:", len(contextual_chunks))
    print("Queries:", len(queries))

    original_results = evaluate(
        original_chunks,
        queries,
        qrels,
    )

    contextual_results = evaluate(
        contextual_chunks,
        queries,
        qrels,
    )

    print("\n=== MEAN METRICS ===")

    for metric in ["precision", "recall", "ndcg"]:
        original = mean_metric(original_results, metric)
        contextual = mean_metric(contextual_results, metric)

        if original is None or contextual is None:
            print(f"{metric}@5: N/A")
            continue

        difference = contextual - original

        print(
            f"{metric}@5: "
            f"Original={original:.3f} | "
            f"Contextual={contextual:.3f} | "
            f"Delta={difference:+.3f}"
        )

    print("\n=== PER-QUERY COMPARISON ===")

    for original, contextual in zip(
        original_results,
        contextual_results,
    ):
        print(f"\n{original['query_id']}")

        print(
            "Original IDs:",
            original["candidate_ids"],
        )
        print(
            "Contextual IDs:",
            contextual["candidate_ids"],
        )

        print(
            "Original grades:",
            original["grades"],
        )
        print(
            "Contextual grades:",
            contextual["grades"],
        )


if __name__ == "__main__":
    main()
