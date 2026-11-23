
"""Lesson 6B.3 — Evidence-grounded multimodal retrieval.

Finds query terms in specific CV profile fields.

Preserves:
- Candidate ID
- Evidence field
- Exact extracted text
- Source PDF
- Extraction method

Deduplicates repeated evidence across source formats.

No OpenAI API calls.

Run:
    python -m scripts.multimodal.demo_multimodal_evidence

    python -m scripts.multimodal.demo_multimodal_evidence \
        --query "Recruitment Analytics Dashboard"
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
    tokenize,
)


OUTPUT_DIR = Path(
    "data/multimodal/retrieval"
)

DEFAULT_QUERY = "Python Docker"

SEARCHABLE_FIELDS = [
    "role",
    "skills",
    "experience",
    "projects",
    "education",
]


def extract_evidence(record, query_terms):
    """Find profile entries containing query terms."""

    profile = record["profile"]

    evidence_items = []

    for field in SEARCHABLE_FIELDS:
        value = profile.get(field)

        if isinstance(value, str):
            entries = [value]

        elif isinstance(value, list):
            entries = [
                item
                for item in value
                if isinstance(item, str)
            ]

        else:
            continue

        for entry in entries:
            entry_terms = set(
                tokenize(entry)
            )

            matched = sorted(
                query_terms & entry_terms
            )

            if not matched:
                continue

            evidence_items.append({
                "field": field,
                "text": entry,
                "matched_terms": matched,
                "source_file": record[
                    "source_file"
                ],
                "extraction_method": record[
                    "extraction_method"
                ],
            })

    return evidence_items


def deduplicate_evidence(items):
    """Merge identical field/text evidence across sources."""

    merged = {}

    for item in items:
        key = (
            item["field"],
            item["text"].casefold().strip(),
        )

        if key not in merged:
            merged[key] = {
                "field": item["field"],
                "text": item["text"],
                "matched_terms": set(),
                "sources": [],
            }

        merged[key]["matched_terms"].update(
            item["matched_terms"]
        )

        source = {
            "source_file": item["source_file"],
            "extraction_method": item[
                "extraction_method"
            ],
        }

        if source not in merged[key]["sources"]:
            merged[key]["sources"].append(
                source
            )

    results = []

    for item in merged.values():
        item["matched_terms"] = sorted(
            item["matched_terms"]
        )

        item["sources"].sort(
            key=lambda source: (
                source["extraction_method"],
                source["source_file"] or "",
            )
        )

        results.append(item)

    results.sort(
        key=lambda item: (
            SEARCHABLE_FIELDS.index(
                item["field"]
            ),
            item["text"].casefold(),
        )
    )

    return results


def search_with_evidence(query, records):
    """Return candidate-level results with evidence."""

    query_terms = set(
        tokenize(query)
    )

    if not query_terms:
        raise ValueError(
            "Search query must not be empty."
        )

    candidate_records = defaultdict(list)

    for record in records:
        candidate_records[
            record["candidate_id"]
        ].append(record)

    results = []

    for candidate_id, sources in (
        candidate_records.items()
    ):
        raw_evidence = []

        for source in sources:
            raw_evidence.extend(
                extract_evidence(
                    source,
                    query_terms,
                )
            )

        evidence = deduplicate_evidence(
            raw_evidence
        )

        matched_terms = set()

        for item in evidence:
            matched_terms.update(
                item["matched_terms"]
            )

        coverage = (
            len(matched_terms)
            / len(query_terms)
        )

        if coverage == 0:
            continue

        results.append({
            "candidate_id": candidate_id,
            "coverage_score": coverage,
            "matched_terms": sorted(
                matched_terms
            ),
            "evidence_count": len(evidence),
            "evidence": evidence,
        })

    results.sort(
        key=lambda result: (
            -result["coverage_score"],
            result["candidate_id"],
        )
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evidence-grounded multimodal CV search"
        )
    )

    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
    )

    args = parser.parse_args()

    print(
        "=== LESSON 6B.3: "
        "EVIDENCE-GROUNDED RETRIEVAL ==="
    )

    records = load_profiles()

    print("Source records:", len(records))
    print("Query:", args.query)

    results = search_with_evidence(
        args.query,
        records,
    )

    print("\nCANDIDATE RESULTS")

    if not results:
        print(
            "No evidence matched the query."
        )

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
            "Coverage score:",
            f"{result['coverage_score']:.3f}",
        )

        print(
            "Matched terms:",
            result["matched_terms"],
        )

        print(
            "Unique evidence items:",
            result["evidence_count"],
        )

        for index, item in enumerate(
            result["evidence"],
            start=1,
        ):
            print(
                f"\n  Evidence {index}"
            )

            print(
                "  Field:",
                item["field"],
            )

            print(
                "  Text:",
                item["text"],
            )

            print(
                "  Matched:",
                item["matched_terms"],
            )

            print(
                "  Supporting sources:",
                len(item["sources"]),
            )

            for source in item["sources"]:
                print(
                    "    -",
                    source["source_file"],
                    f"({source['extraction_method']})",
                )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "evidence_search_results.json"
    )

    output_path.write_text(
        json.dumps(
            {
                "query": args.query,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\nUnique candidate results:",
        len(results),
    )

    print(
        "Saved:",
        output_path,
    )

    print(
        "\n=== EVIDENCE RETRIEVAL COMPLETE ==="
    )


if __name__ == "__main__":
    main()
