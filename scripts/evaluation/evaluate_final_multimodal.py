
"""Lesson 6B.10 — Final multimodal evaluation.

Consolidates saved Phase 6A/6B reports and reruns
the grounded-answer regression tests.

No OpenAI API calls.

Run:
    python -m scripts.evaluation.evaluate_final_multimodal
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
)
from scripts.multimodal.demo_multimodal_evidence import (
    search_with_evidence,
)


BASE = Path("data/multimodal")

CONSISTENCY_PATH = (
    BASE / "reports/hybrid_consistency_evaluation.json"
)

PDF_VALIDATION_PATH = (
    BASE / "retrieval/pdf_evidence_validation.json"
)

VISUAL_PATH = (
    BASE / "reports/visual_evidence_verification.json"
)

NEGATIVE_PATH = (
    BASE / "reports/visual_negative_test_evaluation.json"
)

SAFETY_PATH = (
    BASE / "reports/multimodal_claim_safety_gate.json"
)

GROUNDED_PATH = (
    BASE / "reports/grounded_multimodal_answers.json"
)

OUTPUT_DIR = BASE / "final_evaluation"

JSON_OUTPUT = OUTPUT_DIR / "results.json"
MARKDOWN_OUTPUT = OUTPUT_DIR / "summary.md"


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing report: {path}\n"
            "Complete the corresponding lesson first."
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def get_consistency_metrics(report):
    """Support the consistency report's known summary fields."""

    # Report schemas can differ between lesson versions.
    # Derive matching fields from field-level booleans
    # when the report exposes them.
    def find_field_results(value):
        if isinstance(value, dict):
            for key in (
                "field_matches",
                "field_comparisons",
                "field_results",
            ):
                candidate = value.get(key)

                if isinstance(candidate, dict):
                    booleans = [
                        item
                        for item in candidate.values()
                        if isinstance(item, bool)
                    ]

                    if len(booleans) == 6:
                        return booleans

            for nested in value.values():
                found = find_field_results(nested)

                if found is not None:
                    return found

        return None

    matches = find_field_results(report)

    if matches is not None:
        return {
            "matched_fields": sum(matches),
            "total_fields": len(matches),
            "consistency_score": (
                sum(matches) / len(matches)
            ),
        }

    # Fall back to explicit numeric summary fields.
    def find_number(value, names):
        if not isinstance(value, dict):
            return None

        for name in names:
            number = value.get(name)

            if (
                isinstance(number, (int, float))
                and not isinstance(number, bool)
            ):
                return number

        for nested in value.values():
            found = find_number(nested, names)

            if found is not None:
                return found

        return None

    score = find_number(
        report,
        ("consistency_score", "field_match_score"),
    )

    matched = find_number(
        report,
        ("matched_fields", "matching_fields"),
    )

    total = find_number(
        report,
        ("total_fields", "fields_compared"),
    )

    if (
        score is None
        and matched is None
        and total is None
    ):
        raise ValueError(
            "Could not interpret the consistency report. "
            "Inspect its JSON structure before continuing."
        )

    return {
        "matched_fields": matched,
        "total_fields": total,
        "consistency_score": score,
    }


def count_pdf_validation(report):
    """Count PDF validation decisions from saved records."""

    counts = {
        "verified_pdf_text": 0,
        "vision_profile_only": 0,
        "unsupported": 0,
    }

    for query in report["queries"]:
        for candidate in query["candidates"]:
            for evidence in candidate["evidence"]:
                for validation in evidence["validations"]:
                    status = validation["status"]

                    if status not in counts:
                        raise ValueError(
                            f"Unknown PDF status: {status}"
                        )

                    counts[status] += 1

    return counts


def evaluate_visual(report):
    """Summarize model-based visual verification."""

    results = report["results"]

    counts = {
        "supported": 0,
        "unsupported": 0,
        "uncertain": 0,
    }

    for result in results:
        status = result["status"]

        if status not in counts:
            raise ValueError(
                f"Unknown visual status: {status}"
            )

        counts[status] += 1

    return {
        "total_evidence": len(results),
        **counts,
    }


def evaluate_negative(report):
    """Recompute negative-test metrics from case records."""

    cases = report["evaluation"]["results"]

    correct = sum(
        case["expected_status"]
        == case["predicted_status"]
        for case in cases
    )

    negatives = [
        case
        for case in cases
        if case["expected_status"] == "unsupported"
    ]

    false_acceptances = sum(
        case["predicted_status"] == "supported"
        for case in negatives
    )

    return {
        "total_tests": len(cases),
        "correct": correct,
        "accuracy": (
            correct / len(cases)
            if cases else 0.0
        ),
        "negative_tests": len(negatives),
        "negative_rejections": sum(
            case["predicted_status"] == "unsupported"
            for case in negatives
        ),
        "false_acceptances": false_acceptances,
        "false_acceptance_rate": (
            false_acceptances / len(negatives)
            if negatives else 0.0
        ),
    }


def evaluate_safety(report):
    """Recompute safety-gate test results."""

    tests = report["results"]

    passed = sum(
        test["expected"] == test["actual"]
        for test in tests
    )

    return {
        "total_tests": len(tests),
        "passed": passed,
        "failed": len(tests) - passed,
    }


def evaluate_grounded(report):
    """Check the three controlled demonstration queries."""

    expected = {
        "Python Docker": {
            "candidate": "M001",
            "minimum_evidence": 4,
        },
        "Recruitment Analytics Dashboard": {
            "candidate": "M001",
            "minimum_evidence": 2,
        },
        "Kubernetes Terraform": {
            "candidate": None,
            "minimum_evidence": 0,
        },
    }

    checks = []

    for query, expectation in expected.items():
        matching = [
            item
            for item in report
            if item["query"] == query
        ]

        if len(matching) != 1:
            checks.append({
                "query": query,
                "passed": False,
                "reason": (
                    "Missing or duplicate query result."
                ),
            })
            continue

        item = matching[0]

        eligible = [
            candidate
            for candidate in item["evaluations"]
            if candidate["eligible"]
        ]

        expected_candidate = expectation["candidate"]

        if expected_candidate is None:
            passed = (
                len(eligible) == 0
                and "No candidates have sufficient "
                    "verified evidence"
                    in item["answer"]
            )

        else:
            passed = (
                len(eligible) == 1
                and eligible[0]["candidate_id"]
                    == expected_candidate
                and len(
                    eligible[0]["accepted_evidence"]
                ) >= expectation["minimum_evidence"]
                and all(
                    evidence["decision"] == "accept"
                    for evidence in eligible[0][
                        "accepted_evidence"
                    ]
                )
                and all(
                    evidence["source_file"] in item["answer"]
                    and (
                        f"page {evidence['page_number']}"
                        in item["answer"]
                    )
                    for evidence in eligible[0][
                        "accepted_evidence"
                    ]
                )
            )

        checks.append({
            "query": query,
            "passed": passed,
        })

    return {
        "total_queries": len(checks),
        "passed": sum(
            item["passed"]
            for item in checks
        ),
        "checks": checks,
    }


def evaluate_retrieval():
    """Check basic candidate deduplication behavior."""

    records = load_profiles()

    checks = []

    for query, expected_ids in [
        ("Python Docker", ["M001"]),
        ("Recruitment Analytics Dashboard", ["M001"]),
        ("Kubernetes Terraform", []),
    ]:
        candidates = search_with_evidence(
            query,
            records,
        )

        actual_ids = [
            item["candidate_id"]
            for item in candidates
        ]

        checks.append({
            "query": query,
            "expected_candidate_ids": expected_ids,
            "actual_candidate_ids": actual_ids,
            "passed": actual_ids == expected_ids,
        })

    return {
        "source_records": len(records),
        "unique_candidates": len({
            record["candidate_id"]
            for record in records
        }),
        "total_checks": len(checks),
        "passed": sum(
            check["passed"]
            for check in checks
        ),
        "checks": checks,
    }


def run_regression_tests():
    """Rerun the 15 grounded-answer tests."""

    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_multimodal_grounded_answers.py",
        "-v",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    output = (
        result.stdout + "\n" + result.stderr
    ).strip()

    print("\nREGRESSION TEST OUTPUT")
    print(output)

    import re

    match = re.search(
        r"Ran (\d+) tests?",
        output,
    )

    count = (
        int(match.group(1))
        if match else None
    )

    return {
        "command": " ".join(command),
        "return_code": result.returncode,
        "tests_run": count,
        "passed": (
            result.returncode == 0
            and count == 15
            and re.search(r"(?m)^OK$", output)
                is not None
        ),
    }


def make_markdown(results):
    """Create a portfolio-friendly Markdown summary."""

    consistency = results["profile_consistency"]
    pdf = results["pdf_provenance"]
    visual = results["visual_verification"]
    negative = results["negative_testing"]
    safety = results["claim_safety"]
    grounded = results["grounded_answers"]
    retrieval = results["retrieval"]
    tests = results["regression_tests"]

    return f"""# CV Intelligence — Phase 6B Evaluation

Generated: {results['generated_at']}

## Scope

This evaluation uses one synthetic candidate (M001)
represented by a text PDF and a scanned PDF.

The results measure controlled extraction, retrieval,
verification, and claim-safety behavior. They are not
real-world recruitment accuracy estimates.

## Results

| Component | Result |
|---|---|
| Profile consistency | {consistency['matched_fields']}/{consistency['total_fields']} matching fields |
| PDF text evidence matches | {pdf['verified_pdf_text']} |
| Vision-profile-only references | {pdf['vision_profile_only']} |
| Visual verification | {visual['supported']}/{visual['total_evidence']} supported |
| Negative tests | {negative['correct']}/{negative['total_tests']} correct |
| Fabricated claims rejected | {negative['negative_rejections']}/{negative['negative_tests']} |
| False acceptances | {negative['false_acceptances']} |
| Claim safety tests | {safety['passed']}/{safety['total_tests']} passed |
| Grounded answer queries | {grounded['passed']}/{grounded['total_queries']} passed |
| Retrieval checks | {retrieval['passed']}/{retrieval['total_checks']} passed |
| Regression tests | {tests['tests_run']} run; passed: {tests['passed']} |

## Architecture

1. Hybrid text/vision PDF ingestion.
2. Structured candidate profile extraction.
3. Candidate deduplication across PDF representations.
4. Field-level evidence retrieval.
5. PDF page provenance validation.
6. Model-based visual evidence verification.
7. Deterministic claim safety checks.
8. Evidence-grounded answer rendering.

## Safety behavior

- Claims with incorrect candidate IDs are rejected.
- Claims from mismatched source PDFs are rejected.
- Evidence absent from the source profile is rejected.
- Missing or uncertain visual verification causes abstention.
- Unsupported expertise claims are rejected.
- Only accepted evidence is included in positive answers.

## Limitations

- One synthetic candidate and two PDF representations.
- The two PDFs contain equivalent underlying content.
- The scanned-PDF check uses a vision model and is not
  independent ground-truth certification.
- Only the first page of the scanned PDF was verified.
- The negative-test set contains three fabricated claims.
- The current pipeline relies on saved verification JSON
  rather than authenticated, immutable evidence records.
- The answer renderer trusts the accepted-evidence list
  supplied by the validation pipeline.
- CV statements do not independently establish professional
  experience, identity, education, or skill proficiency.
- Results cannot be generalized to real-world hiring.

## Conclusion

The controlled Phase 6B tests demonstrate that CV
Intelligence can retrieve evidence from multimodal-derived
profiles, apply source-specific claim checks, and produce
answers that cite accepted CV statements.

Broader document coverage, stronger provenance controls,
and independent evaluation are required before any
production recruitment use.
"""


def main():
    print(
        "=== LESSON 6B.10: "
        "FINAL MULTIMODAL EVALUATION ==="
    )

    consistency = get_consistency_metrics(
        read_json(CONSISTENCY_PATH)
    )

    pdf = count_pdf_validation(
        read_json(PDF_VALIDATION_PATH)
    )

    visual = evaluate_visual(
        read_json(VISUAL_PATH)
    )

    negative = evaluate_negative(
        read_json(NEGATIVE_PATH)
    )

    safety = evaluate_safety(
        read_json(SAFETY_PATH)
    )

    grounded = evaluate_grounded(
        read_json(GROUNDED_PATH)
    )

    retrieval = evaluate_retrieval()

    tests = run_regression_tests()

    results = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "scope": (
            "One synthetic candidate with "
            "text and scanned PDF representations"
        ),
        "profile_consistency": consistency,
        "pdf_provenance": pdf,
        "visual_verification": visual,
        "negative_testing": negative,
        "claim_safety": safety,
        "grounded_answers": grounded,
        "retrieval": retrieval,
        "regression_tests": tests,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_OUTPUT.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    MARKDOWN_OUTPUT.write_text(
        make_markdown(results),
        encoding="utf-8",
    )

    print("\nFINAL EVALUATION SUMMARY")

    print(
        "Profile consistency:",
        f"{consistency['matched_fields']}/"
        f"{consistency['total_fields']}",
    )

    print(
        "PDF text evidence matches:",
        pdf["verified_pdf_text"],
    )

    print(
        "Vision-profile references:",
        pdf["vision_profile_only"],
    )

    print(
        "Visual evidence supported:",
        f"{visual['supported']}/"
        f"{visual['total_evidence']}",
    )

    print(
        "Negative tests correct:",
        f"{negative['correct']}/"
        f"{negative['total_tests']}",
    )

    print(
        "Fabricated claims rejected:",
        f"{negative['negative_rejections']}/"
        f"{negative['negative_tests']}",
    )

    print(
        "False acceptances:",
        negative["false_acceptances"],
    )

    print(
        "Claim safety tests:",
        f"{safety['passed']}/"
        f"{safety['total_tests']}",
    )

    print(
        "Grounded answer queries:",
        f"{grounded['passed']}/"
        f"{grounded['total_queries']}",
    )

    print(
        "Retrieval checks:",
        f"{retrieval['passed']}/"
        f"{retrieval['total_checks']}",
    )

    print(
        "Regression tests:",
        tests["tests_run"],
    )

    print(
        "Regression suite passed:",
        tests["passed"],
    )

    print("\nSaved JSON:", JSON_OUTPUT)
    print("Saved Markdown:", MARKDOWN_OUTPUT)

    all_passed = (
        consistency["matched_fields"] == 6
        and consistency["total_fields"] == 6
        and pdf["verified_pdf_text"] == 6
        and pdf["vision_profile_only"] == 6
        and visual["supported"] == 6
        and visual["total_evidence"] == 6
        and negative["correct"] == 6
        and negative["total_tests"] == 6
        and negative["negative_rejections"] == 3
        and negative["false_acceptances"] == 0
        and safety["passed"] == 8
        and safety["total_tests"] == 8
        and grounded["passed"] == 3
        and retrieval["passed"] == 3
        and tests["passed"]
    )

    print(
        "\nAll evaluation checks passed:",
        all_passed,
    )

    if not all_passed:
        raise SystemExit(
            "\n=== FINAL EVALUATION NEEDS REVIEW ==="
        )

    print(
        "\n=== PHASE 6B EVALUATION PASSED ==="
    )


if __name__ == "__main__":
    main()
