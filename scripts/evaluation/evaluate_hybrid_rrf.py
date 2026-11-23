
"""Lesson 4C: Evaluate BM25, Jina Late Chunking, and Hybrid RRF.

Uses:
    data/public_benchmark/chunks.jsonl
    data/public_benchmark/queries.jsonl
    data/public_benchmark/qrels.tsv
    data/public_benchmark/jina_embedding_cache/

Evaluation is candidate-level, before LLM grading.
"""

import csv
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from scripts.agentic_rag.demo_hybrid_agentic_rag import (
    RETRIEVAL_POOL,
    RRF_K,
    candidate_ranking,
    embed_query,
    load_chunks,
    load_late_vectors,
    normalize,
    reciprocal_rank_fusion,
    tokenize,
)


DATA_DIR = Path("data/public_benchmark")
QUERIES_PATH = DATA_DIR / "queries.jsonl"
QRELS_PATH = DATA_DIR / "qrels.tsv"

TOP_K = 5
RELEVANT_GRADE = 2


def load_queries():
    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_qrels():
    qrels = {}

    with QRELS_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if not row:
                continue

            if row[0] == "query_id":
                continue

            query_id, candidate_id, relevance = row[:3]

            qrels.setdefault(query_id, {})[
                candidate_id
            ] = int(relevance)

    return qrels


def precision_at_k(ranking, grades, k=5):
    top = ranking[:k]

    relevant = sum(
        grades.get(candidate_id, 0) >= RELEVANT_GRADE
        for candidate_id in top
    )

    return relevant / k


def recall_at_k(ranking, grades, k=5):
    total_relevant = sum(
        grade >= RELEVANT_GRADE
        for grade in grades.values()
    )

    if total_relevant == 0:
        return 0.0

    retrieved_relevant = sum(
        grades.get(candidate_id, 0) >= RELEVANT_GRADE
        for candidate_id in ranking[:k]
    )

    return retrieved_relevant / total_relevant


def dcg_at_k(relevances, k=5):
    return sum(
        (2 ** grade - 1) / np.log2(rank + 2)
        for rank, grade in enumerate(relevances[:k])
    )


def ndcg_at_k(ranking, grades, k=5):
    actual = [
        grades.get(candidate_id, 0)
        for candidate_id in ranking[:k]
    ]

    ideal = sorted(
        grades.values(),
        reverse=True,
    )[:k]

    ideal_dcg = dcg_at_k(ideal, k)

    if ideal_dcg == 0:
        return 0.0

    return dcg_at_k(actual, k) / ideal_dcg


def evaluate_ranking(ranking, grades):
    return {
        "precision": precision_at_k(ranking, grades),
        "recall": recall_at_k(ranking, grades),
        "ndcg": ndcg_at_k(ranking, grades),
    }


def mean_metrics(rows):
    return {
        metric: float(np.mean([
            row[metric]
            for row in rows
        ]))
        for metric in ("precision", "recall", "ndcg")
    }


def print_metrics(label, metrics):
    print(
        f"{label:<15}"
        f"P@5={metrics['precision']:.3f}  "
        f"R@5={metrics['recall']:.3f}  "
        f"nDCG@5={metrics['ndcg']:.3f}"
    )


def main():
    load_dotenv()

    chunks = load_chunks()
    queries = load_queries()
    qrels = load_qrels()

    print("=== LESSON 4C: HYBRID RETRIEVAL BENCHMARK ===")
    print("Chunks:", len(chunks))
    print("Queries:", len(queries))
    print("Judgment sets:", len(qrels))

    print("\nLoading cached Jina Late Chunking vectors...")
    late_vectors = load_late_vectors(chunks)
    print("Vector shape:", late_vectors.shape)

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    all_results = {
        "BM25": [],
        "Jina Late": [],
        "Hybrid RRF": [],
    }

    for query in queries:
        query_id = query["query_id"]
        query_text = query["text"]
        grades = qrels[query_id]

        print(f"\n=== {query_id}: {query_text} ===")

        # BM25 candidate ranking.
        bm25_scores = bm25.get_scores(
            tokenize(query_text)
        )

        bm25_ranked, _ = candidate_ranking(
            bm25_scores,
            chunks,
            RETRIEVAL_POOL,
        )

        # Jina semantic candidate ranking.
        query_vector = normalize(
            embed_query(query_text).reshape(1, -1)
        )[0]

        jina_scores = late_vectors @ query_vector

        jina_ranked, _ = candidate_ranking(
            jina_scores,
            chunks,
            RETRIEVAL_POOL,
        )

        # Reciprocal Rank Fusion.
        hybrid_ranked, _ = reciprocal_rank_fusion(
            bm25_ranked,
            jina_ranked,
        )

        rankings = {
            "BM25": bm25_ranked,
            "Jina Late": jina_ranked,
            "Hybrid RRF": hybrid_ranked,
        }

        for method, ranking in rankings.items():
            metrics = evaluate_ranking(
                ranking,
                grades,
            )

            all_results[method].append(metrics)

            print_metrics(method, metrics)
            print("  Top 5:", ranking[:TOP_K])

    print("\n" + "=" * 60)
    print("MEAN RESULTS ACROSS ALL QUERIES")
    print("=" * 60)

    for method, rows in all_results.items():
        print_metrics(
            method,
            mean_metrics(rows),
        )

    bm25_mean = mean_metrics(all_results["BM25"])
    hybrid_mean = mean_metrics(all_results["Hybrid RRF"])

    difference = (
        hybrid_mean["ndcg"] - bm25_mean["ndcg"]
    )

    print("\nHybrid vs BM25 nDCG@5 difference:")
    print(f"{difference:+.3f}")

    if difference > 0:
        print("Hybrid improved mean nDCG@5.")
    elif difference < 0:
        print("BM25 performed better on mean nDCG@5.")
    else:
        print("Both methods achieved equal mean nDCG@5.")


if __name__ == "__main__":
    main()
