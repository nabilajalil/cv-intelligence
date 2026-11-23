
from pathlib import Path

from scripts.evaluation.evaluate_chunk_bm25 import (
    load_chunks,
    retrieve_candidates,
)
from scripts.evaluation.evaluate_public_bm25 import (
    load_queries,
    load_qrels,
    ndcg_at_k,
)


CONTEXTUAL_PATH = Path(
    "data/public_benchmark/llm_contextual_chunks.jsonl"
)

TOP_K = 5


def evaluate(chunks, queries, qrels):
    candidate_ids = {
        chunk["candidate_id"]
        for chunk in chunks
    }

    results = []

    for query in queries:
        query_id = query["query_id"]

        # Only evaluate candidates included in this experiment.
        labels = {
            candidate_id: grade
            for candidate_id, grade in qrels[query_id].items()
            if candidate_id in candidate_ids
        }

        retrieved = retrieve_candidates(
            query=query["text"],
            chunks=chunks,
            top_k=TOP_K,
        )

        grades = [
            labels.get(candidate_id, 0)
            for candidate_id in retrieved
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
            "retrieved": retrieved,
            "grades": grades,
            "precision": precision,
            "recall": recall,
            "ndcg": ndcg,
            "total_relevant": total_relevant,
        })

    return results


def mean_metric(results, metric):
    values = [
        result[metric]
        for result in results
        if result[metric] is not None
    ]

    return (
        sum(values) / len(values)
        if values else None
    )


def main():
    contextual_chunks = load_chunks(CONTEXTUAL_PATH)

    if not contextual_chunks:
        raise ValueError("No LLM-contextual chunks found.")

    selected_ids = {
        chunk["candidate_id"]
        for chunk in contextual_chunks
    }

    original_chunks = [
        chunk
        for chunk in load_chunks()
        if chunk["candidate_id"] in selected_ids
    ]

    original_ids = {
        chunk["chunk_id"]
        for chunk in original_chunks
    }

    contextual_ids = [
        chunk["chunk_id"]
        for chunk in contextual_chunks
    ]

    if (
        len(contextual_ids) != len(set(contextual_ids))
        or set(contextual_ids) != original_ids
    ):
        raise ValueError(
            "Original and contextual chunk IDs do not match."
        )

    queries = load_queries()
    qrels = load_qrels()

    print("=== LLM CONTEXTUAL BM25 COMPARISON ===")
    print("Candidates:", len(selected_ids))
    print("Original chunks:", len(original_chunks))
    print("LLM-contextual chunks:", len(contextual_chunks))
    print("Queries:", len(queries))

    original_results = evaluate(
        original_chunks, queries, qrels
    )

    contextual_results = evaluate(
        contextual_chunks, queries, qrels
    )

    print("\n=== MEAN METRICS ===")

    for metric in ["precision", "recall", "ndcg"]:
        original = mean_metric(original_results, metric)
        contextual = mean_metric(contextual_results, metric)

        if original is None or contextual is None:
            print(f"{metric}@5: N/A")
            continue

        print(
            f"{metric}@5: "
            f"Original={original:.3f} | "
            f"LLM Contextual={contextual:.3f} | "
            f"Delta={contextual - original:+.3f}"
        )

    print("\n=== PER-QUERY RESULTS ===")

    for original, contextual in zip(
        original_results,
        contextual_results,
    ):
        print(f"\n{original['query_id']}")
        print(
            "Full matches in subset:",
            original["total_relevant"],
        )
        print("Original IDs:", original["retrieved"])
        print("LLM IDs:", contextual["retrieved"])
        print("Original grades:", original["grades"])
        print("LLM grades:", contextual["grades"])


if __name__ == "__main__":
    main()
