
"""Lesson 6B.7 — Multimodal claim safety gate.

Deterministic checks for claims supported by a
previously generated visual verification report.

No OpenAI API calls.

Run:
    python -m scripts.multimodal.multimodal_claim_safety_gate
"""

import json
import re
from pathlib import Path


PROFILE_PATH = Path(
    "data/multimodal/hybrid_profiles/"
    "synthetic_scanned_cv_profile.json"
)

VISUAL_REPORT_PATH = Path(
    "data/multimodal/reports/"
    "visual_evidence_verification.json"
)

NEGATIVE_REPORT_PATH = Path(
    "data/multimodal/reports/"
    "visual_negative_test_evaluation.json"
)

OUTPUT_PATH = Path(
    "data/multimodal/reports/"
    "multimodal_claim_safety_gate.json"
)


def normalize(text):
    """Normalize evidence for exact-content comparison."""

    if not isinstance(text, str):
        return ""

    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            text.casefold(),
        )
    )


def load_json(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required report: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def field_contains(profile, field, text):
    """Check that evidence belongs to the given field."""

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
        return False

    target = normalize(text)

    return bool(target) and any(
        normalize(item) == target
        for item in entries
    )


def make_claim(
    candidate_id,
    field,
    text,
    source_file,
    page_number=1,
    claim_type="cv_mention",
):
    """Create a proposed claim with provenance."""

    return {
        "candidate_id": candidate_id,
        "claim_type": claim_type,
        "field": field,
        "evidence_text": text,
        "source_file": source_file,
        "page_number": page_number,
    }


def build_verification_lookup(report):
    """Index visual decisions by their full provenance."""

    lookup = {}

    for result in report.get("results", []):
        key = (
            result.get("field"),
            normalize(result.get("text")),
            result.get("source_file"),
            result.get("page_number"),
        )

        # Duplicate verification records are ambiguous.
        # Do not silently overwrite one decision.
        if key in lookup:
            raise ValueError(
                "Duplicate visual verification record: "
                f"{key}"
            )

        lookup[key] = result.get("status")

    return lookup


def validate_claim(
    claim,
    source_profile,
    verification_lookup,
):
    """Apply deterministic claim safety rules."""

    profile = source_profile["profile"]

    if claim.get("claim_type") != "cv_mention":
        return {
            "decision": "reject",
            "reason": "Claim type is not permitted.",
        }

    if (
        claim.get("candidate_id")
        != profile.get("candidate_id")
    ):
        return {
            "decision": "reject",
            "reason": "Candidate ID mismatch.",
        }

    if (
        claim.get("source_file")
        != source_profile.get("source_file")
    ):
        return {
            "decision": "reject",
            "reason": "Source PDF mismatch.",
        }

    field = claim.get("field")
    evidence_text = claim.get("evidence_text")

    if not field_contains(
        profile,
        field,
        evidence_text,
    ):
        return {
            "decision": "reject",
            "reason": (
                "Evidence is absent from the "
                "claimed source profile and field."
            ),
        }

    key = (
        field,
        normalize(evidence_text),
        claim.get("source_file"),
        claim.get("page_number"),
    )

    status = verification_lookup.get(key)

    if status == "supported":
        return {
            "decision": "accept",
            "reason": (
                "Evidence matches the source profile "
                "and has a supported visual decision."
            ),
        }

    if status == "unsupported":
        return {
            "decision": "reject",
            "reason": (
                "Visual verification marked "
                "the evidence unsupported."
            ),
        }

    if status == "uncertain":
        return {
            "decision": "abstain",
            "reason": (
                "Visual verification is uncertain."
            ),
        }

    return {
        "decision": "abstain",
        "reason": (
            "No matching visual verification record."
        ),
    }


def build_test_claims(source_profile, visual_report):
    """Build positive and adversarial test claims."""

    candidate_id = source_profile[
        "profile"
    ]["candidate_id"]

    source_file = source_profile[
        "source_file"
    ]

    positive_record = next(
        result
        for result in visual_report["results"]
        if result["status"] == "supported"
        and result["field"] == "skills"
        and normalize(result["text"]) == "python"
    )

    valid_claim = make_claim(
        candidate_id=candidate_id,
        field=positive_record["field"],
        text=positive_record["text"],
        source_file=source_file,
        page_number=positive_record["page_number"],
    )

    return [
        {
            "test_name": "valid_python_claim",
            "expected": "accept",
            "claim": valid_claim,
        },
        {
            "test_name": "wrong_candidate_id",
            "expected": "reject",
            "claim": {
                **valid_claim,
                "candidate_id": "M999",
            },
        },
        {
            "test_name": "wrong_source_pdf",
            "expected": "reject",
            "claim": {
                **valid_claim,
                "source_file": (
                    "data/multimodal/input/"
                    "synthetic_text_cv.pdf"
                ),
            },
        },
        {
            "test_name": "wrong_page_number",
            "expected": "abstain",
            "claim": {
                **valid_claim,
                "page_number": 99,
            },
        },
        {
            "test_name": "wrong_evidence_field",
            "expected": "reject",
            "claim": {
                **valid_claim,
                "field": "education",
            },
        },
        {
            "test_name": "fabricated_kubernetes_claim",
            "expected": "reject",
            "claim": make_claim(
                candidate_id,
                "experience",
                (
                    "Experienced with Kubernetes "
                    "and Terraform."
                ),
                source_file,
            ),
        },
        {
            "test_name": "unverified_sql_claim",
            "expected": "abstain",
            "claim": make_claim(
                candidate_id,
                "skills",
                "SQL",
                source_file,
            ),
        },
        {
            "test_name": "unsupported_expertise_claim",
            "expected": "reject",
            "claim": {
                **valid_claim,
                "claim_type": "expertise",
            },
        },
    ]


def main():
    print(
        "=== LESSON 6B.7: "
        "MULTIMODAL CLAIM SAFETY GATE ==="
    )

    source_profile = load_json(
        PROFILE_PATH
    )

    visual_report = load_json(
        VISUAL_REPORT_PATH
    )

    negative_report = load_json(
        NEGATIVE_REPORT_PATH
    )

    print(
        "Candidate:",
        source_profile["profile"]["candidate_id"],
    )

    print(
        "Source:",
        source_profile["source_file"],
    )

    print(
        "Visual verification records:",
        len(visual_report["results"]),
    )

    print(
        "Negative test cases:",
        len(
            negative_report["evaluation"]["results"]
        ),
    )

    # Confirm the report belongs to this source.
    if (
        visual_report.get("source_file")
        != source_profile.get("source_file")
    ):
        raise ValueError(
            "Visual report and profile source mismatch."
        )

    lookup = build_verification_lookup(
        visual_report
    )

    tests = build_test_claims(
        source_profile,
        visual_report,
    )

    results = []
    passed_count = 0

    print("\nSAFETY GATE TESTS")

    for test in tests:
        outcome = validate_claim(
            test["claim"],
            source_profile,
            lookup,
        )

        passed = (
            outcome["decision"] == test["expected"]
        )

        if passed:
            passed_count += 1

        print(
            f"\n{test['test_name']}: "
            f"{'PASS' if passed else 'FAIL'}"
        )

        print(
            "Expected:",
            test["expected"],
        )

        print(
            "Actual:",
            outcome["decision"],
        )

        print(
            "Reason:",
            outcome["reason"],
        )

        results.append({
            "test_name": test["test_name"],
            "expected": test["expected"],
            "actual": outcome["decision"],
            "passed": passed,
            "reason": outcome["reason"],
        })

    total = len(tests)

    evaluation = {
        "total_tests": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "all_tests_passed": passed_count == total,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            evaluation,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nEVALUATION SUMMARY")

    print(
        "Passed:",
        f"{passed_count}/{total}",
    )

    print(
        "Failed:",
        total - passed_count,
    )

    print(
        "All tests passed:",
        evaluation["all_tests_passed"],
    )

    print("\nSaved:", OUTPUT_PATH)

    if not evaluation["all_tests_passed"]:
        raise SystemExit(
            "\n=== SAFETY GATE TEST FAILED ==="
        )

    print(
        "\n=== LESSON 6B.7 PASSED ==="
    )


if __name__ == "__main__":
    main()
