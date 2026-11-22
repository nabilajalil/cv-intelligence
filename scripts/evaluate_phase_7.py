
"""Phase 7 — Final application integration evaluation.

Evaluates:
- FastAPI endpoint registration
- Protected search through HTTP TestClient
- Evidence provenance and integrity
- Existing HTTP and grounded-answer regression tests

No OpenAI API calls.
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from cv_intelligence.api import app
from scripts.document_fingerprints import sha256_file
from scripts.demo_protected_grounded_answers import (
    PROTECTED_REPORT_PATH,
)
from scripts.multimodal_claim_safety_gate import load_json


OUTPUT_DIR = Path(
    "data/multimodal/final_evaluation"
)

RESULTS_PATH = OUTPUT_DIR / "phase_7_results.json"
SUMMARY_PATH = OUTPUT_DIR / "phase_7_summary.md"

EXPECTED_QUERIES = [
    {
        "query": "Python Docker",
        "required_skills": ["Python", "Docker"],
        "candidate_count": 1,
        "accepted_evidence": 4,
    },
    {
        "query": "Recruitment Analytics Dashboard",
        "required_skills": [],
        "candidate_count": 1,
        "accepted_evidence": 2,
    },
    {
        "query": "Kubernetes Terraform",
        "required_skills": ["Kubernetes", "Terraform"],
        "candidate_count": 0,
        "accepted_evidence": 0,
    },
]


def check_api_routes(client):
    """Verify required endpoints using OpenAPI."""

    response = client.get("/openapi.json")
    response.raise_for_status()

    paths = response.json().get("paths", {})

    expected = {
        "/health": "get",
        "/search": "post",
        "/multimodal/protected-search": "post",
    }

    checks = []

    for path, method in expected.items():
        passed = method in paths.get(path, {})

        checks.append(
            {
                "path": path,
                "method": method.upper(),
                "passed": passed,
            }
        )

    return {
        "passed": all(
            item["passed"] for item in checks
        ),
        "checks": checks,
    }


def check_document_integrity():
    """Compare the current scanned PDF to the protected hash."""

    report = load_json(PROTECTED_REPORT_PATH)

    source_file = Path(report["source_file"])
    expected_hash = report["source_sha256"]

    if not source_file.is_file():
        return {
            "passed": False,
            "status": "source_missing",
            "source_file": str(source_file),
        }

    actual_hash = sha256_file(source_file)

    return {
        "passed": actual_hash == expected_hash,
        "status": (
            "match"
            if actual_hash == expected_hash
            else "mismatch"
        ),
        "source_file": str(source_file),
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
    }


def check_protected_queries(client):
    """Exercise protected search through HTTP requests."""

    results = []

    for case in EXPECTED_QUERIES:
        response = client.post(
            "/multimodal/protected-search",
            json={
                "query": case["query"],
                "required_skills": case[
                    "required_skills"
                ],
            },
        )

        body = (
            response.json()
            if response.status_code == 200
            else {}
        )

        candidates = body.get("candidates", [])

        accepted = sum(
            len(
                candidate.get(
                    "accepted_evidence",
                    [],
                )
            )
            for candidate in candidates
        )

        evidence = [
            item
            for candidate in candidates
            for item in candidate.get(
                "accepted_evidence",
                [],
            )
        ]

        evidence_valid = all(
            item.get("decision") == "accept"
            and item.get("integrity_status") == "match"
            and item.get("page_number") == 1
            and Path(
                item.get("source_file", "")
            ).name == "synthetic_scanned_cv.pdf"
            for item in evidence
        )

        candidate_ids_valid = all(
            candidate.get("candidate_id") == "M001"
            for candidate in candidates
        )

        passed = (
            response.status_code == 200
            and body.get("candidate_count")
            == case["candidate_count"]
            and len(candidates)
            == case["candidate_count"]
            and accepted
            == case["accepted_evidence"]
            and evidence_valid
            and candidate_ids_valid
        )

        results.append(
            {
                "query": case["query"],
                "http_status": response.status_code,
                "candidate_count": len(candidates),
                "accepted_evidence": accepted,
                "evidence_valid": evidence_valid,
                "passed": passed,
            }
        )

    return {
        "passed": all(
            item["passed"] for item in results
        ),
        "queries": results,
        "total_accepted_evidence": sum(
            item["accepted_evidence"]
            for item in results
        ),
    }


def run_regression_tests():
    """Run all 39 Phase 7 and grounded-answer tests."""

    commands = [
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_protected_api",
            "-v",
        ],
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*grounded_answers.py",
            "-v",
        ],
    ]

    expected_counts = [12, 27]
    suites = []

    for command, expected in zip(
        commands,
        expected_counts,
    ):
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        output = result.stdout + result.stderr

        match = re.search(
            r"Ran (\d+) tests? in",
            output,
        )

        count = (
            int(match.group(1))
            if match
            else 0
        )

        passed = (
            result.returncode == 0
            and count == expected
            and re.search(
                r"^OK\s*$",
                output,
                re.MULTILINE,
            )
            is not None
        )

        suites.append(
            {
                "command": " ".join(command[1:]),
                "expected_tests": expected,
                "tests_run": count,
                "passed": passed,
                "return_code": result.returncode,
                "output_tail": output[-2000:],
            }
        )

    return {
        "passed": all(
            suite["passed"]
            for suite in suites
        ),
        "total_tests": sum(
            suite["tests_run"]
            for suite in suites
        ),
        "suites": suites,
    }


def make_markdown(results):
    """Build a portfolio-friendly Markdown summary."""

    routes = results["api_routes"]
    queries = results["protected_queries"]
    integrity = results["document_integrity"]
    regressions = results["regression_tests"]

    status = (
        "PASSED"
        if results["all_checks_passed"]
        else "FAILED"
    )

    lines = [
        "# CV Intelligence — Phase 7 Evaluation",
        "",
        f"**Status:** {status}",
        "",
        f"**Evaluated at:** {results['evaluated_at']}",
        "",
        "## Summary",
        "",
        "| Check | Result |",
        "|---|---|",
        (
            "| FastAPI routes | "
            f"{sum(x['passed'] for x in routes['checks'])}"
            f"/{len(routes['checks'])} |"
        ),
        (
            "| Protected search queries | "
            f"{sum(x['passed'] for x in queries['queries'])}"
            f"/{len(queries['queries'])} |"
        ),
        (
            "| Accepted evidence items | "
            f"{queries['total_accepted_evidence']} |"
        ),
        (
            "| Document fingerprint | "
            f"{integrity['status']} |"
        ),
        (
            "| Automated regression tests | "
            f"{regressions['total_tests']}/39 |"
        ),
        "",
        "## Protected Search Results",
        "",
        "| Query | Candidates | Accepted evidence | Passed |",
        "|---|---:|---:|---|",
    ]

    for item in queries["queries"]:
        lines.append(
            f"| {item['query']} "
            f"| {item['candidate_count']} "
            f"| {item['accepted_evidence']} "
            f"| {item['passed']} |"
        )

    lines.extend(
        [
            "",
            "## Verified Safety Properties",
            "",
            "- Evidence is tied to a scanned PDF fingerprint.",
            "- Accepted evidence includes source-page provenance.",
            "- Unsupported and uncertain claims are withheld.",
            "- Modified or missing PDFs are blocked by the "
            "integrity gate in the regression tests.",
            "- Existing Agentic RAG routes remain registered.",
            "",
            "## Application Integration",
            "",
            "- FastAPI exposes protected multimodal search.",
            "- Streamlit displays eligible candidates, "
            "evidence, and provenance.",
            "- Streamlit was visually checked separately; "
            "this script does not automate a browser.",
            "",
            "## Limitations",
            "",
            "- Evaluation uses one synthetic multimodal candidate.",
            "- Saved verification reports are local artifacts.",
            "- Document fingerprinting does not independently "
            "authenticate the report publisher.",
            "- CV mentions do not establish actual proficiency.",
            "- Production security, fairness, and performance "
            "have not been evaluated.",
            "",
            "## Reproduce",
            "",
            "```bash",
            "python -m scripts.evaluate_phase_7",
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    print(
        "=== LESSON 7.4: FINAL APPLICATION EVALUATION ==="
    )

    client = TestClient(app)

    print("\nChecking FastAPI routes...")
    routes = check_api_routes(client)

    print("Checking protected search queries...")
    queries = check_protected_queries(client)

    print("Checking PDF fingerprint...")
    integrity = check_document_integrity()

    print("Running 39 regression tests...")
    regressions = run_regression_tests()

    all_passed = all(
        [
            routes["passed"],
            queries["passed"],
            integrity["passed"],
            regressions["passed"],
            regressions["total_tests"] == 39,
        ]
    )

    results = {
        "phase": "7",
        "evaluated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "api_routes": routes,
        "protected_queries": queries,
        "document_integrity": integrity,
        "regression_tests": regressions,
        "all_checks_passed": all_passed,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        make_markdown(results),
        encoding="utf-8",
    )

    print("\n=== EVALUATION SUMMARY ===")

    print(
        "FastAPI routes:",
        sum(
            item["passed"]
            for item in routes["checks"]
        ),
        "/3",
    )

    print(
        "Protected queries:",
        sum(
            item["passed"]
            for item in queries["queries"]
        ),
        "/3",
    )

    print(
        "Accepted evidence:",
        queries["total_accepted_evidence"],
    )

    print(
        "PDF fingerprint:",
        integrity["status"],
    )

    print(
        "Regression tests:",
        regressions["total_tests"],
    )

    print(
        "Regression suites passed:",
        regressions["passed"],
    )

    print(
        "All checks passed:",
        all_passed,
    )

    print("\nSaved:", RESULTS_PATH)
    print("Saved:", SUMMARY_PATH)

    if not all_passed:
        raise SystemExit(
            "=== PHASE 7 EVALUATION FAILED ==="
        )

    print(
        "\n=== PHASE 7 EVALUATION PASSED ==="
    )


if __name__ == "__main__":
    main()
