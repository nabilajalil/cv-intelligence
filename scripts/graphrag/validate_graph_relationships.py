
"""Lesson 5C.2: Relationship consistency validation.

Checks LLM-extracted graph relationships before insertion.

Checks:
1. Project-technology relationships have a validated
   candidate-project relationship.
2. Relationship evidence is present and attributable.
3. Repeated job titles are flagged for employment modeling.
4. Original rejected relationships remain rejected.
5. Results are saved separately from the extraction output.

No OpenAI calls.
No changes to the existing NetworkX graph.
"""

import json
from collections import defaultdict
from pathlib import Path

from scripts.retrieval.demo_parent_cv_verification import load_chunks


INPUT_PATH = Path(
    "data/public_benchmark/graph/entity_extraction/results.json"
)

OUTPUT_DIR = Path(
    "data/public_benchmark/graph/relationship_validation"
)

REPORT_PATH = OUTPUT_DIR / "report.json"

MAX_EXCERPT_CHARS = 180


def load_extraction_results():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing extraction results: {INPUT_PATH}\n"
            "Run Lesson 5C first:\n"
            "python -m scripts.graphrag.demo_graph_entity_extraction"
        )

    return json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )


def normalize_name(value):
    """Normalize names for conservative comparisons."""

    return " ".join(
        value.strip().casefold().split()
    )


def build_chunk_lookup():
    chunks = load_chunks()

    return {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }


def find_valid_projects(candidate_id, accepted):
    """Find candidate projects with accepted ownership edges."""

    projects = set()

    for relation in accepted:
        if (
            relation["relation"] == "CONTRIBUTED_TO"
            and relation["source"] == candidate_id
        ):
            projects.add(
                normalize_name(relation["target"])
            )

    return projects


def find_repeated_roles(candidate_id, accepted):
    """Flag repeated job titles.

    A repeated title is not necessarily a duplicate job.
    It may represent employment at multiple companies.
    """

    roles = defaultdict(list)

    for relation in accepted:
        if (
            relation["relation"] == "HELD_ROLE"
            and relation["source"] == candidate_id
        ):
            key = normalize_name(
                relation["target"]
            )

            roles[key].append(relation)

    return {
        role: entries
        for role, entries in roles.items()
        if len(entries) > 1
    }


def check_project_technology(
    candidate_id,
    relation,
    valid_projects,
    chunk_lookup,
):
    """Check project ownership and basic evidence integrity."""

    warnings = []

    project = normalize_name(
        relation["source"]
    )

    if project not in valid_projects:
        warnings.append(
            "ORPHAN_PROJECT: No accepted candidate-project "
            "relationship exists for this project."
        )

    chunk_id = relation["chunk_id"]
    chunk = chunk_lookup.get(chunk_id)

    if chunk is None:
        warnings.append(
            "MISSING_CHUNK: Evidence chunk not found."
        )

    elif chunk["candidate_id"] != candidate_id:
        warnings.append(
            "WRONG_CANDIDATE: Evidence chunk belongs "
            "to another candidate."
        )

    elif relation["quote"] not in chunk["text"]:
        warnings.append(
            "INVALID_QUOTE: Quote is not an exact substring."
        )

    # A technology quotation should explicitly mention
    # the technology it claims to support.
    if normalize_name(relation["target"]) not in normalize_name(
        relation["quote"]
    ):
        warnings.append(
            "TECHNOLOGY_NOT_IN_QUOTE: Target technology "
            "is absent from the evidence quote."
        )

    # Important limitation:
    # This does not establish that the technology belongs
    # to the correct project section.
    warnings.append(
        "REVIEW_PROJECT_CONTEXT: Confirm that the technology "
        "belongs to this particular project."
    )

    return warnings


def validate_candidate(candidate_result, chunk_lookup):
    candidate_id = candidate_result["candidate_id"]

    accepted = candidate_result.get(
        "accepted",
        [],
    )

    rejected = candidate_result.get(
        "rejected",
        [],
    )

    valid_projects = find_valid_projects(
        candidate_id,
        accepted,
    )

    repeated_roles = find_repeated_roles(
        candidate_id,
        accepted,
    )

    eligible_for_review = []
    needs_attention = []

    for relation in accepted:
        warnings = []

        if relation["relation"] == "PROJECT_USED_TECHNOLOGY":
            warnings.extend(
                check_project_technology(
                    candidate_id,
                    relation,
                    valid_projects,
                    chunk_lookup,
                )
            )

        if relation["relation"] == "HELD_ROLE":
            role_key = normalize_name(
                relation["target"]
            )

            if role_key in repeated_roles:
                warnings.append(
                    "REPEATED_ROLE: This job title occurs "
                    "multiple times. Preserve distinct "
                    "employment records."
                )

        record = dict(relation)
        record["consistency_warnings"] = warnings

        # Context-review warnings are informational.
        # Other warnings indicate a structural issue.
        blocking_warnings = [
            warning
            for warning in warnings
            if not warning.startswith(
                "REVIEW_PROJECT_CONTEXT:"
            )
        ]

        record["structurally_eligible"] = (
            len(blocking_warnings) == 0
        )

        if warnings:
            needs_attention.append(record)
        else:
            eligible_for_review.append(record)

    for relation in rejected:
        record = dict(relation)
        record["structurally_eligible"] = False
        record["consistency_warnings"] = [
            "ORIGINAL_REJECTION: Failed Lesson 5C validation."
        ]

        needs_attention.append(record)

    return {
        "candidate_id": candidate_id,
        "original_accepted": len(accepted),
        "original_rejected": len(rejected),
        "eligible_for_review": eligible_for_review,
        "needs_attention": needs_attention,
        "repeated_roles": {
            role: len(entries)
            for role, entries in repeated_roles.items()
        },
    }


def print_candidate_report(result):
    candidate_id = result["candidate_id"]

    print("\n" + "=" * 65)
    print("Candidate:", candidate_id)
    print("=" * 65)

    print(
        "Originally accepted:",
        result["original_accepted"],
    )

    print(
        "Originally rejected:",
        result["original_rejected"],
    )

    print(
        "No consistency warnings:",
        len(result["eligible_for_review"]),
    )

    print(
        "Relationships needing attention:",
        len(result["needs_attention"]),
    )

    for record in result["needs_attention"]:
        print(
            f"\n{record['source']} "
            f"--[{record['relation']}]--> "
            f"{record['target']}"
        )

        print(
            "  Chunk:",
            record["chunk_id"],
        )

        print(
            "  Structurally eligible:",
            record["structurally_eligible"],
        )

        for warning in record["consistency_warnings"]:
            print("  WARNING:", warning)

    if result["repeated_roles"]:
        print("\nRepeated job titles:")

        for role, count in result["repeated_roles"].items():
            print(
                f"  {role}: {count} occurrences"
            )


def main():
    print(
        "=== LESSON 5C.2: GRAPH RELATIONSHIP VALIDATION ==="
    )

    extraction_results = load_extraction_results()
    chunk_lookup = build_chunk_lookup()

    print(
        "Candidate extraction records:",
        len(extraction_results),
    )

    print(
        "Available source chunks:",
        len(chunk_lookup),
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports = []

    for candidate_result in extraction_results:
        report = validate_candidate(
            candidate_result,
            chunk_lookup,
        )

        reports.append(report)

        print_candidate_report(report)

    total_accepted = sum(
        report["original_accepted"]
        for report in reports
    )

    total_rejected = sum(
        report["original_rejected"]
        for report in reports
    )

    total_attention = sum(
        len(report["needs_attention"])
        for report in reports
    )

    total_no_warnings = sum(
        len(report["eligible_for_review"])
        for report in reports
    )

    summary = {
        "candidates": len(reports),
        "originally_accepted": total_accepted,
        "originally_rejected": total_rejected,
        "no_consistency_warnings": total_no_warnings,
        "needs_attention": total_attention,
    }

    output = {
        "summary": summary,
        "candidates": reports,
    }

    REPORT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved report:", REPORT_PATH)
    print("\n=== LESSON 5C.2 COMPLETE ===")


if __name__ == "__main__":
    main()

