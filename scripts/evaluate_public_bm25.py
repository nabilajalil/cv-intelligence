
import csv
import json
import math
from pathlib import Path

from scripts.retrieval.search_public_cvs import (
    load_public_cvs,
    search_public_cvs,
)


DATA_DIR = Path("data/public_benchmark")
TOP_K = 5


def load_queries():
    with (DATA_DIR / "queries.jsonl").open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_qrels():
    qrels = {}

    with (DATA_DIR / "qrels.tsv").open(
        encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            qrels.setdefault(row["query_id"], {})[
                row["candidate_id"]
            ] = int(row["relevance"])

    return qrels


def ndcg_at_k(grades, all_grades, k):
    def dcg(values):
        return sum(
            (2 ** grade - 1) / math.log2(rank + 2)
            for rank, grade in enumerate(values[:k])
        )

    ideal = dcg(sorted(all_grades, reverse=True))

    return dcg(grades) / ideal if ideal > 0 else None


def evaluate_query(query, documents, qrels, top_k=TOP_K):
    results = search_public_cvs(
        query=query["text"],
        documents=documents,
        top_k=top_k,
    )

    labels = qrels[query["query_id"]]

    grades = [
        labels.get(result["filename"], 0)
        for result in results
    ]

    full_matches = sum(grade == 2 for grade in labels.values())
    retrieved_full = sum(grade == 2 for grade in grades)

    precision = retrieved_full / top_k

    recall = (
        retrieved_full / full_matches
        if full_matches > 0 else None
    )

    ndcg = ndcg_at_k(
        grades=grades,
        all_grades=list(labels.values()),
        k=top_k,
    )

    return {
        "precision": precision,
        "recall": recall,
        "ndcg": ndcg,
        "grades": grades,
        "retrieved": [
            result["filename"]
            for result in results
        ],
    }


def main():
    documents = load_public_cvs()
    queries = load_queries()
    qrels = load_qrels()

    print("=== PUBLIC CV BM25 EVALUATION ===")
    print("CVs:", len(documents))
    print("Queries:", len(queries))
    print("Top-k:", TOP_K)

    results = []

    for query in queries:
        result = evaluate_query(query, documents, qrels)
        results.append(result)

        print(f"\n{query['query_id']}: {query['text']}")
        print("Retrieved IDs:", result["retrieved"])
        print("Relevance grades:", result["grades"])
        print(f"Precision@5: {result['precision']:.2f}")

        if result["recall"] is None:
            print("Recall@5: N/A")
        else:
            print(f"Recall@5: {result['recall']:.2f}")

        if result["ndcg"] is None:
            print("nDCG@5: N/A")
        else:
            print(f"nDCG@5: {result['ndcg']:.2f}")

    print("\n=== MEAN METRICS ===")

    for metric in ["precision", "recall", "ndcg"]:
        values = [
            result[metric]
            for result in results
            if result[metric] is not None
        ]

        mean = sum(values) / len(values) if values else None

        if mean is not None:
            print(f"Mean {metric}@5: {mean:.3f}")


if __name__ == "__main__":
    main()
