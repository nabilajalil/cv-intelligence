
"""Lesson 4D: Evaluate weighted Reciprocal Rank Fusion.

Compare:
    - BM25 only
    - Equal-weight RRF
    - BM25-heavy RRF (2:1, 3:1, 5:1, 10:1)

Uses the existing 50-CV public benchmark and cached
Jina native Late Chunking vectors.

This evaluates retrieval only, before LLM grading.
"""

import json
from collections import defaultdict
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
    tokenize,
)

from scripts.evaluate_hybrid_rrf import (
    evaluate_ranking,
    load_qrels,
    load_queries,
    mean_metrics,
    print_metrics,
)


OUTPUT_PATH = Path(
    "data/public_benchmark/weighted_rrf_results.json"
)

TOP_K = 5
BM25_WEIGHTS = [1, 2, 3, 5, 10]


def weighted_rrf(
    bm25_ranking,
    jina_ranking,
    bm25_weight,
    jina_weight=1.0,
):
    """Combine candidate rankings with weighted RRF."""

    scores = defaultdict(float)

    for rank, candidate_id in enumerate(
        bm25_ranking,
        start=1,
    ):
        scores[candidate_id] += (
            bm25_weight / (RRF_K + rank)
        )

    for rank, candidate_id in enumerate(
        jina_ranking,
        start=1,
    ):
        scores[candidate_id] += (
            jina_weight / (RRF_K + rank)
        )

    ranked = sorted(
        scores,
        key=lambda candidate_id: (
            -scores[candidate_id],
            candidate_id,
        ),
    )

    return ranked


def main():
    load_dotenv()

    chunks = load_chunks()
    queries = load_queries()
    qrels = load_qrels()

    print("=== LESSON 4D: WEIGHTED RRF EVALUATION ===")
    print("Chunks:", len(chunks))
    print("Queries:", len(queries))
    print("BM25 weights:", BM25_WEIGHTS)

    print("\nLoading cached Jina vectors...")
    late_vectors = load_late_vectors(chunks)

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    method_names = (
        ["BM25 only"]
        + [
            f"RRF {weight}:1"
            for weight in BM25_WEIGHTS
        ]
    )

    all_results = {
        name: []
        for name in method_names
    }

    rankings_by_query = {}

    for query in queries:
        query_id = query["query_id"]
        query_text = query["text"]
        grades = qrels[query_id]

        print(f"\n=== {query_id}: {query_text} ===")

        # Rank candidates with BM25.
        bm25_scores = bm25.get_scores(
            tokenize(query_text)
        )

        bm25_ranked, _ = candidate_ranking(
            bm25_scores,
            chunks,
            RETRIEVAL_POOL,
        )

        # Rank candidates with cached Jina vectors.
        query_vector = normalize(
            embed_query(query_text).reshape(1, -1)
        )[0]

        jina_scores = late_vectors @ query_vector

        jina_ranked, _ = candidate_ranking(
            jina_scores,
            chunks,
            RETRIEVAL_POOL,
        )

        rankings = {
            "BM25 only": bm25_ranked,
        }

        for weight in BM25_WEIGHTS:
            name = f"RRF {weight}:1"

            rankings[name] = weighted_rrf(
                bm25_ranked,
                jina_ranked,
                bm25_weight=weight,
            )

        rankings_by_query[query_id] = {
            name: ranking[:TOP_K]
            for name, ranking in rankings.items()
        }

        for name, ranking in rankings.items():
            metrics = evaluate_ranking(
                ranking,
                grades,
            )

            all_results[name].append(metrics)

            print_metrics(name, metrics)

    print("\n" + "=" * 65)
    print("MEAN RESULTS ACROSS ALL QUERIES")
    print("=" * 65)

    summary = {}

    for name, rows in all_results.items():
        metrics = mean_metrics(rows)
        summary[name] = metrics

        print_metrics(name, metrics)

    bm25_ndcg = summary["BM25 only"]["ndcg"]

    print("\nDifference from BM25-only nDCG@5:")

    for name in method_names[1:]:
        difference = (
            summary[name]["ndcg"] - bm25_ndcg
        )

        print(f"  {name:<12} {difference:+.3f}")

    # This is the best setting on the SAME evaluation set.
    # It is exploratory, not a held-out validation result.
    best_method = max(
        method_names,
        key=lambda name: summary[name]["ndcg"],
    )

    print("\nHighest observed mean nDCG@5:")
    print(
        f"  {best_method}: "
        f"{summary[best_method]['ndcg']:.3f}"
    )

    output = {
        "benchmark": {
            "chunks": len(chunks),
            "queries": len(queries),
            "top_k": TOP_K,
            "rrf_k": RRF_K,
            "retrieval_pool": RETRIEVAL_POOL,
            "bm25_weights": BM25_WEIGHTS,
        },
        "summary": summary,
        "rankings_by_query": rankings_by_query,
        "best_observed_method": best_method,
        "note": (
            "Weights were compared on the evaluation queries. "
            "Results are exploratory and need validation on "
            "separate queries before claiming generalization."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nSaved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
