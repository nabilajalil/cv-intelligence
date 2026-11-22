
"""Lesson 6C.3 — Pre/post fingerprint-protected verification.

Security behavior:
- Hash the PDF before verification.
- Run a verifier callback.
- Hash the PDF after verification.
- Save results only when both hashes match.
- Reject missing or modified documents.
- Never persist failed verification results.

This lesson uses simulated verification callbacks.
No OpenAI API calls.

Run:
    python -m scripts.protected_visual_verification
"""

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.document_fingerprints import (
    SCANNED_PDF,
    sha256_file,
)


OUTPUT_DIR = Path(
    "data/multimodal/provenance"
)

TEST_REPORT_PATH = (
    OUTPUT_DIR / "protected_verification_tests.json"
)

VALID_RECORD_PATH = (
    OUTPUT_DIR / "protected_verification_valid.json"
)


class VerificationSafetyError(Exception):
    """Raised when verification cannot be safely saved."""


def protected_verify(
    pdf_path,
    verifier,
    output_path,
):
    """Run verification with pre/post PDF fingerprint checks.

    The verifier receives the PDF path and returns
    a dictionary containing verification results.

    Only successful, unchanged-document results
    may be persisted.
    """

    pdf_path = Path(pdf_path)
    output_path = Path(output_path)

    if not pdf_path.is_file():
        raise VerificationSafetyError(
            "source_file_missing"
        )

    try:
        hash_before = sha256_file(
            pdf_path
        )
    except OSError as error:
        raise VerificationSafetyError(
            "source_file_unreadable"
        ) from error

    try:
        verification_result = verifier(
            pdf_path
        )
    except Exception as error:
        raise VerificationSafetyError(
            "verifier_failed"
        ) from error

    if not isinstance(
        verification_result,
        dict,
    ):
        raise VerificationSafetyError(
            "invalid_verification_result"
        )

    try:
        hash_after = sha256_file(
            pdf_path
        )
    except FileNotFoundError as error:
        raise VerificationSafetyError(
            "source_file_missing_after_verification"
        ) from error
    except OSError as error:
        raise VerificationSafetyError(
            "source_file_unreadable_after_verification"
        ) from error

    if hash_before != hash_after:
        raise VerificationSafetyError(
            "document_changed_during_verification"
        )

    record = {
        "verification_type": (
            "pre_post_fingerprint_protected"
        ),
        "source_file": str(
            pdf_path
        ),
        "hash_algorithm": "sha256",
        "source_sha256": hash_before,
        "verified_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "verification_result": (
            verification_result
        ),
    }

    # Write a temporary file first, then atomically
    # replace the destination after successful checks.
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=".verification_",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(
                temporary_file.name
            )

            json.dump(
                record,
                temporary_file,
                indent=2,
                ensure_ascii=False,
            )

            temporary_file.flush()
            os.fsync(
                temporary_file.fileno()
            )

        # Recheck immediately before publication.
        if sha256_file(pdf_path) != hash_before:
            raise VerificationSafetyError(
                "document_changed_before_publication"
            )

        os.replace(
            temporary_path,
            output_path,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()

    return record


def simulated_verifier(pdf_path):
    """Return a synthetic verification decision."""

    return {
        "model": "simulated_verifier",
        "page_number": 1,
        "results": [
            {
                "evidence_id": "E001",
                "field": "skills",
                "text": "Python",
                "status": "supported",
            }
        ],
    }


def modifying_verifier(pdf_path):
    """Simulate a verifier that changes the PDF."""

    with Path(pdf_path).open("ab") as file:
        file.write(
            b"\n% Modification during verification\n"
        )

    return simulated_verifier(
        pdf_path
    )


def failing_verifier(pdf_path):
    """Simulate an unexpected verifier failure."""

    raise RuntimeError(
        "Simulated vision model failure."
    )


def invalid_verifier(pdf_path):
    """Return an invalid verification result."""

    return [
        "This is not a verification dictionary."
    ]


def run_test(
    name,
    pdf_path,
    verifier,
    output_path,
    expected_outcome,
    expected_error=None,
):
    """Execute a scenario and check its outcome."""

    outcome = None
    error_code = None

    try:
        protected_verify(
            pdf_path,
            verifier,
            output_path,
        )

        outcome = "saved"

    except VerificationSafetyError as error:
        outcome = "blocked"
        error_code = str(error)

    passed = (
        outcome == expected_outcome
    )

    if expected_error is not None:
        passed = (
            passed
            and error_code == expected_error
        )

    if expected_outcome == "blocked":
        passed = (
            passed
            and not Path(output_path).exists()
        )

    if expected_outcome == "saved":
        passed = (
            passed
            and Path(output_path).is_file()
        )

    print(
        f"{name}: "
        f"{outcome} "
        f"({error_code or 'no error'}) "
        f"=> {'PASS' if passed else 'FAIL'}"
    )

    return {
        "name": name,
        "expected_outcome": expected_outcome,
        "actual_outcome": outcome,
        "expected_error": expected_error,
        "actual_error": error_code,
        "passed": passed,
    }


def main():
    print(
        "=== LESSON 6C.3: "
        "PROTECTED VISUAL VERIFICATION ==="
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove an earlier successful demonstration
    # record so the test checks this run's output.
    if VALID_RECORD_PATH.exists():
        VALID_RECORD_PATH.unlink()

    results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # 1. Unchanged source document.
        results.append(
            run_test(
                "01 Unchanged PDF",
                SCANNED_PDF,
                simulated_verifier,
                VALID_RECORD_PATH,
                "saved",
            )
        )

        # 2. Document modified during verification.
        modified_pdf = (
            temp_dir / "modified.pdf"
        )

        modified_pdf.write_bytes(
            SCANNED_PDF.read_bytes()
        )

        results.append(
            run_test(
                "02 PDF modified during verification",
                modified_pdf,
                modifying_verifier,
                temp_dir / "modified_result.json",
                "blocked",
                "document_changed_during_verification",
            )
        )

        # 3. Missing source document.
        results.append(
            run_test(
                "03 Missing PDF",
                temp_dir / "missing.pdf",
                simulated_verifier,
                temp_dir / "missing_result.json",
                "blocked",
                "source_file_missing",
            )
        )

        # 4. Verifier throws an exception.
        results.append(
            run_test(
                "04 Verifier failure",
                SCANNED_PDF,
                failing_verifier,
                temp_dir / "failed_result.json",
                "blocked",
                "verifier_failed",
            )
        )

        # 5. Verifier returns an invalid type.
        results.append(
            run_test(
                "05 Invalid verifier output",
                SCANNED_PDF,
                invalid_verifier,
                temp_dir / "invalid_result.json",
                "blocked",
                "invalid_verification_result",
            )
        )

    passed = sum(
        item["passed"]
        for item in results
    )

    report = {
        "lesson": "6C.3",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_tests": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
        "verifier_type": "simulated",
        "original_pdf_modified": False,
    }

    TEST_REPORT_PATH.write_text(
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
        TEST_REPORT_PATH,
    )

    if VALID_RECORD_PATH.is_file():
        print(
            "Saved:",
            VALID_RECORD_PATH,
        )

    if passed != len(results):
        raise SystemExit(
            "Protected verification tests failed."
        )

    print(
        "\n=== LESSON 6C.3 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
