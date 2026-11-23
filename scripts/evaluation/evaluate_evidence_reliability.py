
"""Lesson 4I: Evidence classification reliability evaluation.

Runs the Lesson 4H evidence assessor multiple times
on the same candidates and compares its outputs.

Measures:
- Per-skill classification agreement
- Citation validation failures
- Candidate ranking stability

Also exports a human-review template.

Important: Agreement is not accuracy.
"""

import csv
import json
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from scripts.retrieval.demo_evidence_aware_ranking import (
    LEVEL_SCORES,
    assess_candidate,
    rank_bm25,
)

from scripts.retrieval.demo_parent_cv_verification import (
    collect_candidate_chunks,
    extract_required_skills,
    find_skill_evidence,
    load_chunks,
)


QUERY = "Find candidates with Python and SQL"

RUNS = 3
CANDIDATE_LIMIT = 5

OUTPUT_DIR = Path(
    "data/public_benchmark/evidence_reliability"
)

LEVELS = [
    "professional_use",
    "project_use",
    "skill_listing",
    "learning_only",
    "unclear",
    "no_evidence",
]


def select_candidates(chunks, parents, required_skills):
    """Choose a fixed set of verified BM25 candidates."""

    ranking, bm25_scores = rank_bm25(
        chunks,
        QUERY,
    )

    selected = []

    for candidate_id in ranking:
        evidence = find_skill_evidence(
            parents[candidate_id],
            required_skills,
        )

        if all(
            evidence.get(skill)
            for skill in required_skills
        ):
            selected.append(candidate_id)

        if len(selected) >= CANDIDATE_LIMIT:
            break

    return selected, bm25_scores


def evaluate_one_run(
    llm,
    candidate_ids,
    parents,
    required_skills,
    bm25_scores,
):
    results = []

    for candidate_id in candidate_ids:
        print(f"  Assessing {candidate_id}...")

        try:
            assessments = assess_candidate(
                llm,
                candidate_id,
                parents[candidate_id],
                required_skills,
            )

            scores = [
                item["score"]
                for item in assessments
            ]

            results.append({
                "candidate_id": candidate_id,
                "assessments": assessments,
                "minimum_skill_score": min(scores),
                "total_evidence_score": sum(scores),
                "bm25_score": bm25_scores[candidate_id],
            })

        except Exception as error:
            print(
                f"  ERROR for {candidate_id}: "
                f"{type(error).__name__}: {error}"
            )

            # Failed assessments remain visible and
            # receive a conservative zero score.
            results.append({
                "candidate_id": candidate_id,
                "assessments": [
                    {
                        "skill": skill,
                        "validated_level": "unclear",
                        "score": 0,
                        "warnings": [
                            f"Assessment failed: {error}"
                        ],
                    }
                    for skill in required_skills
                ],
                "minimum_skill_score": 0,
                "total_evidence_score": 0,
                "bm25_score": bm25_scores[candidate_id],
            })

    results.sort(
        key=lambda item: (
            -item["minimum_skill_score"],
            -item["total_evidence_score"],
            -item["bm25_score"],
            item["candidate_id"],
        )
    )

    return results


def build_label_lookup(results):
    lookup = {}

    for candidate in results:
        candidate_id = candidate["candidate_id"]

        for assessment in candidate["assessments"]:
            key = (
                candidate_id,
                assessment["skill"],
            )

            lookup[key] = assessment["validated_level"]

    return lookup


def compare_runs(all_runs, candidate_ids, required_skills):
    """Calculate agreement across repeated classifications."""

    lookups = [
        build_label_lookup(run)
        for run in all_runs
    ]

    rows = []
    stable_count = 0

    for candidate_id in candidate_ids:
        for skill in required_skills:
            key = (candidate_id, skill)

            labels = [
                lookup.get(key, "unclear")
                for lookup in lookups
            ]

            stable = len(set(labels)) == 1

            if stable:
                stable_count += 1

            rows.append({
                "candidate_id": candidate_id,
                "skill": skill,
                "labels": labels,
                "stable": stable,
            })

    total = len(rows)

    agreement = (
        stable_count / total
        if total
        else 0.0
    )

    return {
        "total_skill_cases": total,
        "stable_cases": stable_count,
        "unstable_cases": total - stable_count,
        "exact_agreement": agreement,
        "details": rows,
    }


def citation_statistics(all_runs):
    """Count missing or invalid evidence quotations."""

    total = 0
    invalid_skill = 0
    invalid_context = 0
    warnings = 0

    for run in all_runs:
        for candidate in run:
            for item in candidate["assessments"]:
                total += 1

                level = item.get(
                    "original_level",
                    item.get("validated_level"),
                )

                if level in (
                    "professional_use",
                    "project_use",
                    "skill_listing",
                    "learning_only",
                ):
                    if not item.get(
                        "skill_quote_valid",
                        False,
                    ):
                        invalid_skill += 1

                if level in (
                    "professional_use",
                    "project_use",
                ):
                    if not item.get(
                        "context_quote_valid",
                        False,
                    ):
                        invalid_context += 1

                if item.get("warnings"):
                    warnings += 1

    return {
        "assessments": total,
        "invalid_skill_quotes": invalid_skill,
        "invalid_context_quotes": invalid_context,
        "assessments_with_warnings": warnings,
    }


def export_review_template(
    candidate_ids,
    required_skills,
    all_runs,
):
    """Export examples for manual evidence-label review."""

    path = OUTPUT_DIR / "human_review.csv"

    first_lookup = {}

    for candidate in all_runs[0]:
        for assessment in candidate["assessments"]:
            first_lookup[
                (
                    candidate["candidate_id"],
                    assessment["skill"],
                )
            ] = assessment

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "candidate_id",
            "skill",
            "model_label",
            "skill_quote",
            "context_quote",
            "human_label",
            "review_notes",
        ])

        for candidate_id in candidate_ids:
            for skill in required_skills:
                item = first_lookup.get(
                    (candidate_id, skill),
                    {},
                )

                writer.writerow([
                    candidate_id,
                    skill,
                    item.get("validated_level", ""),
                    item.get("skill_quote", ""),
                    item.get("context_quote", ""),
                    "",
                    "",
                ])

    return path


def main():
    load_dotenv()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = load_chunks()
    parents = collect_candidate_chunks(chunks)
    required_skills = extract_required_skills(QUERY)

    candidate_ids, bm25_scores = select_candidates(
        chunks,
        parents,
        required_skills,
    )

    print("=== LESSON 4I: EVIDENCE RELIABILITY ===")
    print("Query:", QUERY)
    print("Required skills:", required_skills)
    print("Candidates:", candidate_ids)
    print("Repeated runs:", RUNS)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    all_runs = []

    for run_number in range(1, RUNS + 1):
        print(
            f"\n=== RUN {run_number}/{RUNS} ==="
        )

        results = evaluate_one_run(
            llm,
            candidate_ids,
            parents,
            required_skills,
            bm25_scores,
        )

        all_runs.append(results)

        print(
            "Ranking:",
            [
                item["candidate_id"]
                for item in results
            ],
        )

    agreement = compare_runs(
        all_runs,
        candidate_ids,
        required_skills,
    )

    citations = citation_statistics(all_runs)

    rankings = [
        [
            item["candidate_id"]
            for item in run
        ]
        for run in all_runs
    ]

    ranking_stable = all(
        ranking == rankings[0]
        for ranking in rankings[1:]
    )

    print("\n=== CLASSIFICATION RELIABILITY ===")

    print(
        "Exact agreement:",
        f"{agreement['exact_agreement']:.1%}",
    )

    print(
        "Stable skill cases:",
        agreement["stable_cases"],
        "/",
        agreement["total_skill_cases"],
    )

    print(
        "Unstable skill cases:",
        agreement["unstable_cases"],
    )

    print("\n=== UNSTABLE CLASSIFICATIONS ===")

    for row in agreement["details"]:
        if not row["stable"]:
            print(
                row["candidate_id"],
                row["skill"],
                "->",
                row["labels"],
            )

    print("\n=== CITATION VALIDATION ===")

    for key, value in citations.items():
        print(f"{key}: {value}")

    print("\n=== RANKING STABILITY ===")
    print("All rankings identical:", ranking_stable)

    for index, ranking in enumerate(
        rankings,
        start=1,
    ):
        print(f"Run {index}: {ranking}")

    review_path = export_review_template(
        candidate_ids,
        required_skills,
        all_runs,
    )

    output = {
        "query": QUERY,
        "required_skills": required_skills,
        "candidate_ids": candidate_ids,
        "runs": all_runs,
        "classification_agreement": agreement,
        "citation_statistics": citations,
        "rankings": rankings,
        "ranking_stable": ranking_stable,
        "limitations": [
            "Agreement is not accuracy.",
            "Only five candidates and one query are tested.",
            "Human labels have not yet been provided.",
            "Quotation validation does not prove that "
            "the classification is semantically correct.",
        ],
    }

    results_path = OUTPUT_DIR / "results.json"

    results_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved results:", results_path)
    print("Human review template:", review_path)


if __name__ == "__main__":
    main()
