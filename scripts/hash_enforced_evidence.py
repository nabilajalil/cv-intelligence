
"""Lesson 6C.2 — Hash-enforced multimodal evidence.

Security rules:
1. Evidence must be bound to a source SHA-256.
2. The current source PDF must match that SHA-256.
3. Candidate identity and source must match.
4. The existing claim safety gate must accept the claim.
5. Missing or modified PDFs fail closed.

No OpenAI API calls.

Run:
    python -m scripts.hash_enforced_evidence
"""

import copy
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.document_fingerprints import (
    BOUND_EVIDENCE_OUTPUT,
    SCANNED_PDF,
    sha256_file,
)

from scripts.multimodal.multimodal_claim_safety_gate import (
    PROFILE_PATH,
    load_json,
    make_claim,
    normalize,
    validate_claim,
)


OUTPUT_PATH = Path(
    "data/multimodal/provenance/"
    "hash_enforcement_tests.json"
)


def make_result(decision, reason, integrity_status):
    """Return a consistent safety decision."""

    return {
        "decision": decision,
        "reason": reason,
        "integrity_status": integrity_status,
    }


def build_bound_lookup(bound_report):
    """Build a verification lookup from hash-bound records."""

    lookup = {}

    for item in bound_report["results"]:
        key = (
            item["field"],
            normalize(item["text"]),
            item["source_file"],
            item["page_number"],
        )

        if key in lookup:
            raise ValueError(
                f"Duplicate evidence record: {key}"
            )

        if (
            item.get("source_sha256")
            != bound_report["source_sha256"]
        ):
            raise ValueError(
                "Evidence fingerprint does not match "
                "its parent report."
            )

        lookup[key] = item["status"]

    return lookup


def validate_hash_enforced_claim(
    claim,
    source_profile,
    bound_report,
    pdf_path=None,
):
    """Validate source integrity before claim evidence.

    pdf_path is an explicit document location used
    for testing or controlled document resolution.

    The caller must supply the actual trusted source
    path in a production integration.
    """

    expected_source = bound_report.get(
        "source_file"
    )

    expected_sha256 = bound_report.get(
        "source_sha256"
    )

    if (
        not isinstance(expected_source, str)
        or not expected_source
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in expected_sha256
        )
    ):
        return make_result(
            "reject",
            "invalid_fingerprint_binding",
            "invalid",
        )

    if bound_report.get("hash_algorithm") != "sha256":
        return make_result(
            "reject",
            "unsupported_hash_algorithm",
            "invalid",
        )

    if (
        claim.get("source_file")
        != expected_source
        or source_profile.get("source_file")
        != expected_source
    ):
        return make_result(
            "reject",
            "source_binding_mismatch",
            "invalid",
        )

    # A document resolver can supply a different
    # physical location for the same logical source,
    # such as a temporary file during testing.
    actual_path = (
        Path(pdf_path)
        if pdf_path is not None
        else Path(expected_source)
    )

    if not actual_path.is_file():
        return make_result(
            "reject",
            "source_file_missing",
            "missing",
        )

    try:
        actual_sha256 = sha256_file(
            actual_path
        )
    except OSError:
        return make_result(
            "reject",
            "source_file_unreadable",
            "error",
        )

    if actual_sha256 != expected_sha256:
        return make_result(
            "reject",
            "document_fingerprint_mismatch",
            "mismatch",
        )

    try:
        lookup = build_bound_lookup(
            bound_report
        )
    except (ValueError, KeyError, TypeError):
        return make_result(
            "reject",
            "invalid_evidence_binding",
            "match",
        )

    # Reuse the existing Phase 6B claim safety gate.
    result = validate_claim(
        claim,
        source_profile,
        lookup,
    )

    return {
        **result,
        "integrity_status": "match",
    }


def create_python_claim(source_profile):
    """Create a known visually supported claim."""

    return make_claim(
        candidate_id=source_profile[
            "profile"
        ]["candidate_id"],
        field="skills",
        text="Python",
        source_file=source_profile[
            "source_file"
        ],
        page_number=1,
        claim_type="cv_mention",
    )


def run_case(
    name,
    claim,
    source_profile,
    bound_report,
    expected_decision,
    expected_reason=None,
    pdf_path=None,
):
    """Run and score a single test case."""

    result = validate_hash_enforced_claim(
        claim,
        source_profile,
        bound_report,
        pdf_path=pdf_path,
    )

    passed = (
        result["decision"]
        == expected_decision
    )

    if expected_reason is not None:
        passed = (
            passed
            and result["reason"]
                == expected_reason
        )

    print(
        f"{name}: "
        f"{result['decision']} "
        f"({result['reason']}) "
        f"=> {'PASS' if passed else 'FAIL'}"
    )

    return {
        "name": name,
        "expected_decision": expected_decision,
        "expected_reason": expected_reason,
        "actual": result,
        "passed": passed,
    }


def main():
    print(
        "=== LESSON 6C.2: "
        "HASH-ENFORCED EVIDENCE ==="
    )

    source_profile = load_json(
        PROFILE_PATH
    )

    bound_report = load_json(
        BOUND_EVIDENCE_OUTPUT
    )

    claim = create_python_claim(
        source_profile
    )

    results = []

    # 1. Original source and valid claim.
    results.append(
        run_case(
            "01 Valid document and claim",
            claim,
            source_profile,
            bound_report,
            "accept",
        )
    )

    # 2. Wrong candidate identity.
    wrong_candidate = copy.deepcopy(
        claim
    )

    wrong_candidate[
        "candidate_id"
    ] = "M999"

    results.append(
        run_case(
            "02 Wrong candidate",
            wrong_candidate,
            source_profile,
            bound_report,
            "reject",
        )
    )

    # 3. Wrong source binding.
    wrong_source = copy.deepcopy(
        claim
    )

    wrong_source["source_file"] = (
        "data/multimodal/input/"
        "synthetic_text_cv.pdf"
    )

    results.append(
        run_case(
            "03 Wrong source",
            wrong_source,
            source_profile,
            bound_report,
            "reject",
            "source_binding_mismatch",
        )
    )

    # 4. Fabricated skill.
    fabricated_claim = copy.deepcopy(
        claim
    )

    fabricated_claim[
        "evidence_text"
    ] = "Kubernetes"

    results.append(
        run_case(
            "04 Fabricated skill",
            fabricated_claim,
            source_profile,
            bound_report,
            "reject",
        )
    )

    # 5. Real profile mention but not verified.
    unverified_claim = copy.deepcopy(
        claim
    )

    unverified_claim[
        "evidence_text"
    ] = "SQL"

    results.append(
        run_case(
            "05 Unverified SQL",
            unverified_claim,
            source_profile,
            bound_report,
            "abstain",
        )
    )

    # 6. Modified PDF in a temporary directory.
    # Never modify the original source document.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        modified_pdf = (
            temp_dir / "modified_cv.pdf"
        )

        modified_pdf.write_bytes(
            SCANNED_PDF.read_bytes()
        )

        with modified_pdf.open("ab") as file:
            file.write(
                b"\n% Intentional test modification\n"
            )

        results.append(
            run_case(
                "06 Modified PDF",
                claim,
                source_profile,
                bound_report,
                "reject",
                "document_fingerprint_mismatch",
                pdf_path=modified_pdf,
            )
        )

        # 7. Missing PDF.
        missing_pdf = (
            temp_dir / "does_not_exist.pdf"
        )

        results.append(
            run_case(
                "07 Missing PDF",
                claim,
                source_profile,
                bound_report,
                "reject",
                "source_file_missing",
                pdf_path=missing_pdf,
            )
        )

        # 8. Unmodified copy at another location.
        unchanged_pdf = (
            temp_dir / "unchanged_cv.pdf"
        )

        unchanged_pdf.write_bytes(
            SCANNED_PDF.read_bytes()
        )

        results.append(
            run_case(
                "08 Unmodified PDF copy",
                claim,
                source_profile,
                bound_report,
                "accept",
                pdf_path=unchanged_pdf,
            )
        )

    # 9. Tampered fingerprint binding.
    bad_binding = copy.deepcopy(
        bound_report
    )

    bad_binding["source_sha256"] = (
        "0" * 64
    )

    results.append(
        run_case(
            "09 Tampered fingerprint",
            claim,
            source_profile,
            bad_binding,
            "reject",
            "document_fingerprint_mismatch",
        )
    )

    # 10. Unsupported hash algorithm.
    wrong_algorithm = copy.deepcopy(
        bound_report
    )

    wrong_algorithm[
        "hash_algorithm"
    ] = "md5"

    results.append(
        run_case(
            "10 Unsupported hash algorithm",
            claim,
            source_profile,
            wrong_algorithm,
            "reject",
            "unsupported_hash_algorithm",
        )
    )

    passed = sum(
        item["passed"]
        for item in results
    )

    report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "lesson": "6C.2",
        "source_file": str(
            SCANNED_PDF
        ),
        "binding_type": bound_report.get(
            "binding_type"
        ),
        "total_tests": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(
        "Tests passed:",
        f"{passed}/{len(results)}",
    )
    print(
        "Tests failed:",
        len(results) - passed,
    )

    print(
        "\nSaved:",
        OUTPUT_PATH,
    )

    if passed != len(results):
        raise SystemExit(
            "Hash-enforcement tests failed."
        )

    print(
        "\n=== LESSON 6C.2 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
