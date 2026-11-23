
"""Lesson 5H.1: Final GraphRAG benchmark evaluation.

Compares:
1. BM25 retrieval
2. BM25 + graph skill verification
3. BM25 + graph + deterministic claim validation

Uses existing benchmark qrels and graph artifacts.
No OpenAI API calls are required.

This is an in-sample benchmark, not held-out validation.
"""

import csv
import json
from pathlib import Path

from scripts.demo_parent_cv_verification import (
    load_chunks,
    collect_candidate_chunks,
)

from scripts.evaluate_graph_hybrid_retrieval import (
    build_bm25,
    rank_candidates_bm25,
    extract_skills,
    graph_verified_candidates,
    filter_ranking,
    load_graph,
)

from scripts.evaluate_hybrid_rrf import (
    load_queries,
    load_qrels,
    evaluate_ranking,
)

from scripts.graphrag.demo_deterministic_graph_answers import (
    collect_evidence_registry,
    compute_skill_coverage,
)

from scripts.graphrag.demo_validated_graph_answers import (
    normalize_retrieval_query,
)

from scripts.graphrag.demo_privacy_safe_graph_answers import (
    make_safe_registry,
)

from scripts.graphrag.demo_claim_validated_graph_answers import (
    build_claims,
    validate_all_claims,
)


TOP_K = 5

OUTPUT_DIR = Path(
    "data/public_benchmark/graph/final_evaluation"
)


def normalize_query_record(record):
    """Support the benchmark's existing query schema."""

    query_id = record["query_id"]

    query_text = record.get(
        "text",
        record.get("query", ""),
    )

    return query_id, query_text


def mean(values):
    if not values:
        return 0.0

    return sum(values) / len(values)


def summarize(rows, method):
    matching = [
        row
        for row in rows
        if row["method"] == method
    ]

    return {
        "precision": mean([
            row["precision"]
            for row in matching
        ]),
        "recall": mean([
            row["recall"]
            for row in matching
        ]),
        "ndcg": mean([
            row["ndcg"]
            for row in matching
        ]),
    }


def main():
    print(
        "=== LESSON 5H.1: "
        "FINAL GRAPHRAG EVALUATION ==="
    )

    chunks = load_chunks()
    graph = load_graph()

    candidate_chunks = collect_candidate_chunks(
        chunks
    )

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    queries = load_queries()
    qrels = load_qrels()

    bm25 = build_bm25(chunks)

    print("CV chunks:", len(chunks))
    print("Candidates:", len(candidate_chunks))
    print("Queries:", len(queries))
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    methods = [
        "BM25",
        "BM25+Graph",
        "BM25+Graph+Claims",
    ]

    ranking_rows = []
    claim_rows = []

    for record in queries:
        query_id, query_text = normalize_query_record(
            record
        )

        required_skills = extract_skills(
            query_text
        )

        retrieval_query = normalize_retrieval_query(
            query_text
        )

        print("\n" + "=" * 60)
        print("Query:", query_id)
        print("Text:", query_text)
        print("Required skills:", required_skills)

        bm25_ranking = rank_candidates_bm25(
            retrieval_query,
            chunks,
            bm25,
        )

        verified = graph_verified_candidates(
            graph,
            required_skills,
        )

        graph_ranking = filter_ranking(
            bm25_ranking,
            verified,
        )

        rankings = {
            "BM25": bm25_ranking,
            "BM25+Graph": graph_ranking,
            "BM25+Graph+Claims": graph_ranking,
        }

        grades = qrels[query_id]

        for method in methods:
            ranking = rankings[method]

            metrics = evaluate_ranking(
                ranking,
                grades,
            )

            ranking_rows.append({
                "query_id": query_id,
                "method": method,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "ndcg": metrics["ndcg"],
                "top5": ranking[:TOP_K],
            })

            print(
                f"{method:22s}",
                f"P@5={metrics['precision']:.3f}",
                f"R@5={metrics['recall']:.3f}",
                f"nDCG@5={metrics['ndcg']:.3f}",
            )

        selected = graph_ranking[:TOP_K]

        registry, candidate_map = (
            collect_evidence_registry(
                selected,
                required_skills,
                candidate_chunks,
                graph,
                chunk_lookup,
            )
        )

        coverage = compute_skill_coverage(
            selected,
            required_skills,
            registry,
        )

        safe_registry = make_safe_registry(
            registry,
            chunk_lookup,
        )

        proposed = build_claims(
            selected,
            required_skills,
            safe_registry,
            candidate_map,
        )

        accepted, rejected = validate_all_claims(
            proposed,
            safe_registry,
            selected,
        )

        claim_rows.append({
            "query_id": query_id,
            "selected_candidates": selected,
            "required_skills": required_skills,
            "coverage": coverage,
            "proposed_claims": len(proposed),
            "accepted_claims": len(accepted),
            "rejected_claims": len(rejected),
        })

        print(
            "Skill coverage:",
            f"{coverage['covered']}/{coverage['expected']}",
        )

        print(
            "Claims:",
            f"{len(accepted)} accepted,",
            f"{len(rejected)} rejected",
        )

    print("\n" + "=" * 60)
    print("FINAL RETRIEVAL SUMMARY")
    print("=" * 60)

    summary = {}

    for method in methods:
        metrics = summarize(
            ranking_rows,
            method,
        )

        summary[method] = metrics

        print(
            f"{method:22s}",
            f"P@5={metrics['precision']:.3f}",
            f"R@5={metrics['recall']:.3f}",
            f"nDCG@5={metrics['ndcg']:.3f}",
        )

    total_covered = sum(
        row["coverage"]["covered"]
        for row in claim_rows
    )

    total_expected = sum(
        row["coverage"]["expected"]
        for row in claim_rows
    )

    total_accepted = sum(
        row["accepted_claims"]
        for row in claim_rows
    )

    total_rejected = sum(
        row["rejected_claims"]
        for row in claim_rows
    )

    print("\nFINAL CLAIM SUMMARY")
    print(
        "Mandatory skill coverage:",
        f"{total_covered}/{total_expected}",
    )
    print("Accepted claims:", total_accepted)
    print("Rejected claims:", total_rejected)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = OUTPUT_DIR / "results.json"

    results_path.write_text(
        json.dumps(
            {
                "ranking_results": ranking_rows,
                "claim_results": claim_rows,
                "summary": summary,
                "total_skill_coverage": {
                    "covered": total_covered,
                    "expected": total_expected,
                },
                "total_accepted_claims": total_accepted,
                "total_rejected_claims": total_rejected,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_path = OUTPUT_DIR / "ranking_metrics.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)

        writer.writerow([
            "query_id",
            "method",
            "precision_at_5",
            "recall_at_5",
            "ndcg_at_5",
        ])

        for row in ranking_rows:
            writer.writerow([
                row["query_id"],
                row["method"],
                row["precision"],
                row["recall"],
                row["ndcg"],
            ])

    print("\nSaved:", results_path)
    print("Saved:", csv_path)

    print(
        "\n=== LESSON 5H.1 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
