
"""Lesson 6B.2 — Candidate deduplication.

Groups multiple CV source records by candidate ID.

Preserves:
- Candidate identity
- Text and vision extraction provenance
- Matched query terms
- All supporting source files

Uses lexical query-term coverage for this tiny
experimental corpus.

No API calls.

Run:
    python -m scripts.demo_candidate_deduplication

    python -m scripts.demo_candidate_deduplication \
        --query "Recruitment Analytics Dashboard"
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
    profile_to_text,
    tokenize,
)


OUTPUT_DIR = Path(
    "data/multimodal/retrieval"
)

DEFAULT_QUERY = "Python Docker engineer"


def group_by_candidate(records):
    """Group source records under one candidate ID."""

    groups = defaultdict(list)

    for record in records:
        candidate_id = record["candidate_id"]

        groups[candidate_id].append(record)

    return dict(groups)


def calculate_coverage(query_tokens, document_text):
    """Calculate unique query-term coverage."""

    unique_query_terms = set(query_tokens)

    if not unique_query_terms:
        return 0.0, []

    document_terms = set(
        tokenize(document_text)
    )

    matched_terms = sorted(
        unique_query_terms & document_terms
    )

    score = (
        len(matched_terms)
        / len(unique_query_terms)
    )

    return score, matched_terms


def rank_candidates(query, records):
    """Rank unique candidates, retaining all sources."""

    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError(
            "Please provide a nonempty search query."
        )

    grouped = group_by_candidate(records)

    results = []

    for candidate_id, sources in grouped.items():
        source_results = []
        all_matched_terms = set()

        for source in sources:
            profile_text = profile_to_text(
                source["profile"]
            )

            score, matched_terms = (
                calculate_coverage(
                    query_tokens,
                    profile_text,
                )
            )

            all_matched_terms.update(
                matched_terms
            )

            source_results.append({
                "record_id": source["record_id"],
                "source_file": source["source_file"],
                "extraction_method": source[
                    "extraction_method"
                ],
                "coverage_score": score,
                "matched_terms": matched_terms,
            })

        unique_query_terms = set(
            query_tokens
        )

        candidate_score = (
            len(all_matched_terms)
            / len(unique_query_terms)
        )

        # Use a stable order for source provenance.
        source_results.sort(
            key=lambda item: (
                item["extraction_method"],
                item["source_file"] or "",
            )
        )

        role = next(
            (
                source["profile"].get("role")
                for source in sources
                if source["profile"].get("role")
            ),
            None,
        )

        results.append({
            "candidate_id": candidate_id,
            "role": role,
            "coverage_score": candidate_score,
            "matched_terms": sorted(
                all_matched_terms
            ),
            "source_count": len(sources),
            "sources": source_results,
        })

    # A candidate must match at least one term
    # to appear in the search results.
    results = [
        result
        for result in results
        if result["coverage_score"] > 0
    ]

    results.sort(
        key=lambda item: (
            -item["coverage_score"],
            item["candidate_id"],
        )
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Candidate-level multimodal CV retrieval"
        )
    )

    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Search query",
    )

    args = parser.parse_args()

    print(
        "=== LESSON 6B.2: "
        "CANDIDATE DEDUPLICATION ==="
    )

    records = load_profiles()

    print("Source records:", len(records))

    unique_candidates = {
        record["candidate_id"]
        for record in records
    }

    print(
        "Unique candidates:",
        len(unique_candidates),
    )

    print("Query:", args.query)

    results = rank_candidates(
        args.query,
        records,
    )

    print("\nCANDIDATE RESULTS")

    if not results:
        print("No candidates matched the query.")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(f"\nRank {rank}")
        print(
            "Candidate:",
            result["candidate_id"],
        )
        print(
            "Role:",
            result["role"],
        )
        print(
            "Coverage score:",
            f"{result['coverage_score']:.3f}",
        )
        print(
            "Matched terms:",
            result["matched_terms"],
        )
        print(
            "Supporting sources:",
            result["source_count"],
        )

        for source in result["sources"]:
            print(
                "  -",
                source["source_file"],
                f"({source['extraction_method']})",
            )

    print(
        "\nUnique candidate results:",
        len(results),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "deduplicated_search_results.json"
    )

    output_path.write_text(
        json.dumps(
            {
                "query": args.query,
                "source_record_count": len(records),
                "unique_candidate_count": len(
                    unique_candidates
                ),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output_path)

    print(
        "\n=== DEDUPLICATION COMPLETE ==="
    )


if __name__ == "__main__":
    main()

