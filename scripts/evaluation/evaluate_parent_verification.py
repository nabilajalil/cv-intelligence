
"""Lesson 4F: Evaluate parent CV evidence verification.

Compare:
1. BM25
2. BM25 + full-CV skill verification
3. Weighted RRF 10:1 + full-CV skill verification

Uses the existing public benchmark:
- 224 indexed chunks
- 50 candidate CVs
- 5 recruiter queries
- graded relevance judgments

This script does not call an LLM.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from scripts.agentic_rag.demo_hybrid_agentic_rag import (
    candidate_ranking,
    embed_query,
    load_chunks,
    load_late_vectors,
    normalize,
    tokenize,
)

from scripts.retrieval.demo_parent_cv_verification import (
    collect_candidate_chunks,
    extract_required_skills,
    find_skill_evidence,
    classify_match,
)

from scripts.evaluation.evaluate_weighted_rrf import weighted_rrf

from scripts.evaluation.evaluate_hybrid_rrf import (
    evaluate_ranking,
    load_qrels,
    load_queries,
    mean_metrics,
    print_metrics,
)


OUTPUT_PATH = Path(
    "data/public_benchmark/parent_verification_results.json"
)

TOP_K = 5
BM25_WEIGHT = 10


def verify_candidate_ranking(
    ranking: list[str],
    parent_chunks: dict,
    required_skills: list[str],
) -> tuple[list[str], list[dict]]:
    """Keep candidates with evidence for every required skill.

    Preserve the input ranking order.
    """

    verified_ids = []
    verification_records = []

    for candidate_id in ranking:
        candidate_chunks = parent_chunks.get(
            candidate_id,
            [],
        )

        evidence = find_skill_evidence(
            candidate_chunks,
            required_skills,
        )

        status = classify_match(evidence)

        verification_records.append({
            "candidate_id": candidate_id,
            "status": status,
            "evidence": evidence,
        })

        if status == "FULL MATCH":
            verified_ids.append(candidate_id)

    return verified_ids, verification_records


def main():
    load_dotenv()

    chunks = load_chunks()
    queries = load_queries()
    qrels = load_qrels()

    parents = collect_candidate_chunks(chunks)

    print("=== LESSON 4F: PARENT VERIFICATION BENCHMARK ===")
    print("Chunks:", len(chunks))
    print("Candidates:", len(parents))
    print("Queries:", len(queries))

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    print("\nLoading cached Jina Late Chunking vectors...")
    late_vectors = load_late_vectors(chunks)
    print("Vector shape:", late_vectors.shape)

    methods = [
        "BM25",
        "BM25 + Verify",
        "RRF 10:1 + Verify",
    ]

    all_results = {
        method: []
        for method in methods
    }

    query_results = {}

    for query in queries:
        query_id = query["query_id"]
        query_text = query["text"]

        # Prefer the benchmark's explicit required_skills.
        required_skills = query.get(
            "required_skills",
        ) or extract_required_skills(query_text)

        grades = qrels[query_id]

        print(f"\n=== {query_id}: {query_text} ===")
        print("Required skills:", required_skills)

        # --------------------------------------------------
        # 1. BM25 candidate ranking
        # --------------------------------------------------

        bm25_scores = bm25.get_scores(
            tokenize(query_text)
        )

        # Use all 50 candidates so verification can
        # filter candidates without losing recall
        # due to a small retrieval pool.
        bm25_ranked, _ = candidate_ranking(
            bm25_scores,
            chunks,
            len(parents),
        )

        # --------------------------------------------------
        # 2. Jina Late Chunking candidate ranking
        # --------------------------------------------------

        query_vector = normalize(
            embed_query(query_text).reshape(1, -1)
        )[0]

        jina_scores = late_vectors @ query_vector

        jina_ranked, _ = candidate_ranking(
            jina_scores,
            chunks,
            len(parents),
        )

        # --------------------------------------------------
        # 3. Weighted RRF
        # --------------------------------------------------

        hybrid_ranked = weighted_rrf(
            bm25_ranked,
            jina_ranked,
            bm25_weight=BM25_WEIGHT,
        )

        # --------------------------------------------------
        # 4. Parent CV verification
        # --------------------------------------------------

        bm25_verified, bm25_records = (
            verify_candidate_ranking(
                bm25_ranked,
                parents,
                required_skills,
            )
        )

        hybrid_verified, hybrid_records = (
            verify_candidate_ranking(
                hybrid_ranked,
                parents,
                required_skills,
            )
        )

        rankings = {
            "BM25": bm25_ranked,
            "BM25 + Verify": bm25_verified,
            "RRF 10:1 + Verify": hybrid_verified,
        }

        query_results[query_id] = {
            "query": query_text,
            "required_skills": required_skills,
            "rankings": {
                method: ranking[:TOP_K]
                for method, ranking in rankings.items()
            },
            "verified_candidate_count": {
                "BM25": len(bm25_verified),
                "RRF 10:1": len(hybrid_verified),
            },
            "bm25_verification_records": bm25_records,
            "hybrid_verification_records": hybrid_records,
        }

        for method, ranking in rankings.items():
            metrics = evaluate_ranking(
                ranking,
                grades,
            )

            all_results[method].append(metrics)

            print_metrics(method, metrics)

            print(
                "  Top 5:",
                ranking[:TOP_K],
            )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("MEAN RESULTS ACROSS ALL QUERIES")
    print("=" * 65)

    summary = {}

    for method in methods:
        metrics = mean_metrics(
            all_results[method]
        )

        summary[method] = metrics
        print_metrics(method, metrics)

    print("\nDifference from BM25 baseline:")

    baseline_ndcg = summary["BM25"]["ndcg"]

    for method in methods[1:]:
        difference = (
            summary[method]["ndcg"]
            - baseline_ndcg
        )

        print(
            f"  {method:<22} "
            f"{difference:+.3f}"
        )

    output = {
        "benchmark": {
            "chunks": len(chunks),
            "candidates": len(parents),
            "queries": len(queries),
            "top_k": TOP_K,
            "bm25_weight": BM25_WEIGHT,
        },
        "summary": summary,
        "query_results": query_results,
        "limitations": [
            "Verification checks explicit skill mentions.",
            "Mentions do not prove professional experience.",
            "The relevance labels are skill-based, so this "
            "is not an independent assessment of quality.",
            "The RRF weight was selected using the same "
            "five benchmark queries.",
        ],
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

    print("\nSaved results to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
