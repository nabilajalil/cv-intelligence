
"""Lesson 6C.7 — Final protected multimodal evaluation.

Checks:
- Real protected visual verification results.
- Current PDF fingerprint.
- Protected grounded-answer behavior.
- Modified and missing PDF rejection.
- Combined Phase 6B/6C regression tests.

Outputs:
    data/multimodal/final_evaluation/phase_6c_results.json
    data/multimodal/final_evaluation/phase_6c_summary.md

No OpenAI API calls.

Run:
    python -m scripts.evaluate_phase_6c
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from scripts.document_fingerprints import (
    sha256_file,
)
from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
)
from scripts.demo_protected_grounded_answers import (
    PROTECTED_REPORT_PATH,
    build_protected_context,
    run_query,
)
from scripts.multimodal.multimodal_claim_safety_gate import (
    load_json,
)


OUTPUT_DIR = Path(
    "data/multimodal/final_evaluation"
)

RESULTS_PATH = OUTPUT_DIR / "phase_6c_results.json"
SUMMARY_PATH = OUTPUT_DIR / "phase_6c_summary.md"

PROTECTED_ANSWERS_PATH = Path(
    "data/multimodal/provenance/"
    "protected_grounded_answers.json"
)

EXPECTED_QUERIES = {
    "Python Docker": {
        "eligible": True,
        "accepted": 4,
    },
    "Recruitment Analytics Dashboard": {
        "eligible": True,
        "accepted": 2,
    },
    "Kubernetes Terraform": {
        "eligible": False,
        "accepted": 0,
    },
}


def check_protected_vision():
    """Inspect the real protected vision report."""

    report = load_json(PROTECTED_REPORT_PATH)

    results = report["results"]

    actual_counts = {
        "supported": sum(
            item["status"] == "supported"
            for item in results
        ),
        "unsupported": sum(
            item["status"] == "unsupported"
            for item in results
        ),
        "uncertain": sum(
            item["status"] == "uncertain"
            for item in results
        ),
    }

    counts_match = (
        actual_counts == report["counts"]
    )

    expected_outcome = (
        len(results) == 6
        and actual_counts["supported"] == 6
        and actual_counts["unsupported"] == 0
        and actual_counts["uncertain"] == 0
    )

    source_matches = all(
        item["source_file"]
        == report["source_file"]
        and item["page_number"]
        == report["page_number"]
        for item in results
    )

    passed = all([
        report.get("verification_type")
        == "pre_post_fingerprint_protected",
        report.get("hash_algorithm") == "sha256",
        counts_match,
        expected_outcome,
        source_matches,
    ])

    return {
        "passed": passed,
        "model": report.get("model"),
        "verification_type": report.get(
            "verification_type"
        ),
        "evidence_total": len(results),
        "counts": actual_counts,
        "counts_match": counts_match,
        "source_matches": source_matches,
    }


def check_document_fingerprint():
    """Compare the current PDF with the protected hash."""

    report = load_json(PROTECTED_REPORT_PATH)

    source = Path(report["source_file"])

    expected_hash = report["source_sha256"]

    if not source.is_file():
        return {
            "passed": False,
            "source_file": str(source),
            "status": "missing",
            "expected_sha256": expected_hash,
            "actual_sha256": None,
        }

    try:
        actual_hash = sha256_file(source)
    except OSError as error:
        return {
            "passed": False,
            "source_file": str(source),
            "status": "unreadable",
            "error": str(error),
            "expected_sha256": expected_hash,
            "actual_sha256": None,
        }

    matched = actual_hash == expected_hash

    return {
        "passed": matched,
        "source_file": str(source),
        "status": (
            "match" if matched else "mismatch"
        ),
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
    }


def check_grounded_queries():
    """Rerun protected queries and verify their results."""

    records = load_profiles()
    context = build_protected_context()

    checks = []

    for query, expected in EXPECTED_QUERIES.items():
        result = run_query(
            query,
            records,
            context,
        )

        evaluations = result["evaluations"]

        eligible = any(
            item["eligible"]
            for item in evaluations
        )

        accepted = sum(
            len(item["accepted_evidence"])
            for item in evaluations
        )

        # Accepted evidence must never appear in a
        # query that has no eligible candidates.
        if not eligible:
            answer_is_safe = (
                "No candidates" in result["answer"]
            )
        else:
            answer_is_safe = (
                "Candidates with verified CV mentions:"
                in result["answer"]
            )

        passed = all([
            eligible == expected["eligible"],
            accepted == expected["accepted"],
            answer_is_safe,
        ])

        checks.append({
            "query": query,
            "passed": passed,
            "eligible": eligible,
            "accepted_evidence": accepted,
            "expected_eligible": expected[
                "eligible"
            ],
            "expected_accepted": expected[
                "accepted"
            ],
            "answer_is_safe": answer_is_safe,
        })

    return {
        "passed": all(
            item["passed"] for item in checks
        ),
        "passed_count": sum(
            item["passed"] for item in checks
        ),
        "total": len(checks),
        "checks": checks,
    }


def check_integrity_tests():
    """Inspect the completed 6C.5 tamper tests."""

    report = load_json(
        PROTECTED_ANSWERS_PATH
    )

    tests = report["integrity_tests"]

    expected_names = {
        "modified_pdf",
        "missing_pdf",
    }

    names = [
        item["name"]
        for item in tests
    ]

    tests_pass = all(
        item["passed"]
        and item["accepted_evidence"] == 0
        for item in tests
    )

    passed = all([
        len(tests) == 2,
        set(names) == expected_names,
        len(names) == len(set(names)),
        tests_pass,
        report["integrity_tests_passed"] == 2,
        report["integrity_tests_total"] == 2,
    ])

    return {
        "passed": passed,
        "passed_count": sum(
            item["passed"] for item in tests
        ),
        "total": len(tests),
        "tests": tests,
        "note": (
            "These are saved demonstration results. "
            "The live regression suite independently "
            "reruns modified/missing PDF checks."
        ),
    }


def run_regression_suite():
    """Run the Phase 6B and 6C grounded-answer tests."""

    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*grounded_answers.py",
        "-v",
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    output = (
        process.stdout
        + "\n"
        + process.stderr
    )

    import re

    match = re.search(
        r"Ran (\d+) tests? in",
        output,
    )

    test_count = (
        int(match.group(1))
        if match
        else 0
    )

    passed = (
        process.returncode == 0
        and test_count == 27
        and "OK" in output
    )

    return {
        "passed": passed,
        "test_count": test_count,
        "expected_test_count": 27,
        "return_code": process.returncode,
        "output": output,
    }


def make_markdown(results):
    """Build a readable portfolio evaluation summary."""

    vision = results["protected_vision"]
    fingerprint = results["document_fingerprint"]
    queries = results["grounded_queries"]
    integrity = results["integrity_tests"]
    regression = results["regression_tests"]

    status = (
        "PASS"
        if results["all_checks_passed"]
        else "FAIL"
    )

    lines = [
        "# CV Intelligence — Phase 6C Evaluation",
        "",
        f"**Overall status:** {status}",
        "",
        f"**Generated:** {results['generated_at']}",
        "",
        "## Evaluation Results",
        "",
        "| Check | Result |",
        "|---|---|",
        (
            "| Protected visual verification | "
            f"{vision['counts']['supported']}/"
            f"{vision['evidence_total']} supported |"
        ),
        (
            "| PDF fingerprint | "
            f"{fingerprint['status']} |"
        ),
        (
            "| Grounded-answer queries | "
            f"{queries['passed_count']}/"
            f"{queries['total']} passed |"
        ),
        (
            "| Document-integrity tests | "
            f"{integrity['passed_count']}/"
            f"{integrity['total']} passed |"
        ),
        (
            "| Regression tests | "
            f"{regression['test_count']}/27 passed |"
        ),
        "",
        "## Grounded Query Checks",
        "",
        "| Query | Eligible | Accepted evidence | Pass |",
        "|---|---|---:|---|",
    ]

    for item in queries["checks"]:
        lines.append(
            f"| {item['query']} "
            f"| {item['eligible']} "
            f"| {item['accepted_evidence']} "
            f"| {item['passed']} |"
        )

    lines.extend([
        "",
        "## Safety Properties Demonstrated",
        "",
        "- Verified claims require the expected source PDF fingerprint.",
        "- Modified and missing source PDFs are rejected in controlled tests.",
        "- Unsupported visual evidence is rejected.",
        "- Uncertain visual evidence is withheld.",
        "- Candidate identity and evidence content are checked.",
        "- Answers distinguish CV mentions from independently verified qualifications.",
        "",
        "## Scope and Limitations",
        "",
        "- The evaluation uses a small synthetic multimodal dataset with one candidate.",
        "- The vision verification is model-based, not independent human verification.",
        "- SHA-256 checks document-byte integrity against the saved fingerprint.",
        "- The verification JSON is not cryptographically signed or authenticated.",
        "- The protected workflow is demonstrated separately from the main FastAPI/Streamlit application.",
        "- These results do not establish real-world recruitment accuracy, fairness, or production security.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python -m scripts.demo_protected_grounded_answers",
        "python -m unittest discover -s tests -p 'test_*grounded_answers.py' -v",
        "python -m scripts.evaluate_phase_6c",
        "```",
        "",
    ])

    return "\n".join(lines)


def main():
    print(
        "=== LESSON 6C.7: FINAL EVALUATION ==="
    )

    print("\nChecking protected vision report...")
    vision = check_protected_vision()

    print("Checking current PDF fingerprint...")
    fingerprint = check_document_fingerprint()

    print("Rerunning protected grounded queries...")
    queries = check_grounded_queries()

    print("Checking document-integrity tests...")
    integrity = check_integrity_tests()

    print("Running 27 regression tests...")
    regression = run_regression_suite()

    checks = [
        vision["passed"],
        fingerprint["passed"],
        queries["passed"],
        integrity["passed"],
        regression["passed"],
    ]

    results = {
        "phase": "6C",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "protected_vision": vision,
        "document_fingerprint": fingerprint,
        "grounded_queries": queries,
        "integrity_tests": integrity,
        "regression_tests": regression,
        "all_checks_passed": all(checks),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        make_markdown(results),
        encoding="utf-8",
    )

    print("\nEVALUATION SUMMARY")
    print(
        "Protected visual evidence:",
        f"{vision['counts']['supported']}/"
        f"{vision['evidence_total']}",
    )

    print(
        "PDF fingerprint:",
        fingerprint["status"],
    )

    print(
        "Grounded queries:",
        f"{queries['passed_count']}/"
        f"{queries['total']}",
    )

    print(
        "Integrity tests:",
        f"{integrity['passed_count']}/"
        f"{integrity['total']}",
    )

    print(
        "Regression tests:",
        regression["test_count"],
    )

    print(
        "Regression suite passed:",
        regression["passed"],
    )

    print(
        "All checks passed:",
        results["all_checks_passed"],
    )

    print("\nSaved:", RESULTS_PATH)
    print("Saved:", SUMMARY_PATH)

    if not results["all_checks_passed"]:
        raise SystemExit(
            "\nPHASE 6C EVALUATION FAILED"
        )

    print(
        "\n=== PHASE 6C EVALUATION PASSED ==="
    )


if __name__ == "__main__":
    main()
