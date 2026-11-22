
"""Compare original and enriched CV chunks using BM25."""

import json
from pathlib import Path

from scripts.evaluate_chunk_bm25 import retrieve_candidates
from scripts.evaluate_public_bm25 import ndcg_at_k


DATA_DIR = Path("data/public_benchmark")


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_qrels(path):
    qrels = {}

    with path.open("r", encoding="utf-8") as file:
        header = file.readline().strip().split("\t")

        for line in file:
            values = line.strip().split("\t")
            row = dict(zip(header, values))

            query_id = row["query_id"]
            candidate_id = row["candidate_id"]
            relevance = int(row["relevance"])

            qrels.setdefault(query_id, {})[
                candidate_id
            ] = relevance

    return qrels


def evaluate(chunks, queries, qrels, top_k=5):
    totals = {
        "precision": 0.0,
        "recall": 0.0,
        "ndcg": 0.0,
    }

    for query in queries:
        query_id = query["query_id"]
        query_text = query["text"]

        rankings = retrieve_candidates(
            query_text,
            chunks,
            top_k=top_k,
        )

        # The retrieval function returns ranked candidate IDs.
        candidate_ids = [
            item["candidate_id"]
            if isinstance(item, dict)
            else item
            for item in rankings
        ]

        judgments = qrels[query_id]

        retrieved_grades = [
            judgments.get(candidate_id, 0)
            for candidate_id in candidate_ids
        ]

        all_grades = list(judgments.values())

        relevant_retrieved = sum(
            grade == 2
            for grade in retrieved_grades
        )

        total_relevant = sum(
            grade == 2
            for grade in all_grades
        )

        precision = relevant_retrieved / top_k

        recall = (
            relevant_retrieved / total_relevant
            if total_relevant
            else 0.0
        )

        ndcg = ndcg_at_k(
            retrieved_grades,
            all_grades,
            top_k,
        )

        totals["precision"] += precision
        totals["recall"] += recall
        totals["ndcg"] += ndcg

    count = len(queries)

    return {
        name: value / count
        for name, value in totals.items()
    }


def main():
    original_chunks = load_jsonl(
        DATA_DIR / "chunks.jsonl"
    )

    enriched_chunks = load_jsonl(
        DATA_DIR / "enriched_chunks.jsonl"
    )

    queries = load_jsonl(
        DATA_DIR / "queries.jsonl"
    )

    qrels = load_qrels(
        DATA_DIR / "qrels.tsv"
    )

    print("=== STAGE 3A: BM25 COMPARISON ===")
    print("Queries:", len(queries))
    print("Original chunks:", len(original_chunks))
    print("Enriched chunks:", len(enriched_chunks))

    original = evaluate(
        original_chunks,
        queries,
        qrels,
    )

    enriched = evaluate(
        enriched_chunks,
        queries,
        qrels,
    )

    print("\nMetric          Original     Enriched")
    print("------------------------------------")

    for metric in ("precision", "recall", "ndcg"):
        print(
            f"{metric:<15} "
            f"{original[metric]:.3f}        "
            f"{enriched[metric]:.3f}"
        )


if __name__ == "__main__":
    main()
