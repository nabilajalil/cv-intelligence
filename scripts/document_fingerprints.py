
"""Phase 6C.1 — Document fingerprints and tamper detection.

Create SHA-256 fingerprints for synthetic CV PDFs.
Bind existing visual-verification evidence to the
scanned PDF's fingerprint.

Demonstrate:
1. Original document matches.
2. Modified document is rejected.
3. Original document remains unchanged.

No API calls.

Run:
    python -m scripts.document_fingerprints
"""

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path


TEXT_PDF = Path(
    "data/multimodal/input/synthetic_text_cv.pdf"
)

SCANNED_PDF = Path(
    "data/multimodal/input/synthetic_scanned_cv.pdf"
)

VISUAL_REPORT = Path(
    "data/multimodal/reports/"
    "visual_evidence_verification.json"
)

OUTPUT_DIR = Path(
    "data/multimodal/provenance"
)

FINGERPRINTS_OUTPUT = (
    OUTPUT_DIR / "document_fingerprints.json"
)

BOUND_EVIDENCE_OUTPUT = (
    OUTPUT_DIR / "fingerprint_bound_evidence.json"
)

TAMPER_TEST_OUTPUT = (
    OUTPUT_DIR / "tamper_detection_test.json"
)


def sha256_file(path):
    """Compute SHA-256 from a file's exact bytes."""

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def fingerprint_document(path):
    """Create a fingerprint record for a PDF."""

    path = Path(path)

    return {
        "source_file": str(path),
        "algorithm": "sha256",
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def verify_document(path, expected_sha256):
    """Compare the current file with a stored fingerprint."""

    path = Path(path)

    if not path.is_file():
        return {
            "valid": False,
            "reason": "source_file_missing",
            "actual_sha256": None,
        }

    actual = sha256_file(path)

    if actual != expected_sha256:
        return {
            "valid": False,
            "reason": "document_fingerprint_mismatch",
            "actual_sha256": actual,
        }

    return {
        "valid": True,
        "reason": "document_fingerprint_match",
        "actual_sha256": actual,
    }


def load_visual_report():
    """Load the saved Phase 6B visual-verification report."""

    if not VISUAL_REPORT.is_file():
        raise FileNotFoundError(
            f"Missing visual report: {VISUAL_REPORT}"
        )

    return json.loads(
        VISUAL_REPORT.read_text(
            encoding="utf-8"
        )
    )


def bind_visual_evidence(
    visual_report,
    scanned_fingerprint,
):
    """Bind visual evidence to the exact scanned PDF."""

    report_source = visual_report["source_file"]

    expected_source = scanned_fingerprint[
        "source_file"
    ]

    if (
        Path(report_source).resolve()
        != Path(expected_source).resolve()
    ):
        raise ValueError(
            "Visual report source does not match "
            "the scanned PDF."
        )

    bound_results = []

    for result in visual_report["results"]:
        if (
            Path(result["source_file"]).resolve()
            != Path(expected_source).resolve()
        ):
            raise ValueError(
                "Evidence source does not match "
                "the scanned PDF."
            )

        bound_results.append({
            **result,
            "source_sha256": scanned_fingerprint[
                "sha256"
            ],
        })

    return {
        "binding_type": (
            "post_hoc_fingerprint_binding"
        ),
        "source_file": expected_source,
        "source_sha256": scanned_fingerprint[
            "sha256"
        ],
        "hash_algorithm": "sha256",
        "page_number": visual_report[
            "page_number"
        ],
        "verification_model": visual_report[
            "model"
        ],
        "evidence_count": len(bound_results),
        "results": bound_results,
    }


def test_tamper_detection(
    source_pdf,
    expected_sha256,
):
    """Modify a temporary copy, never the original."""

    original_check = verify_document(
        source_pdf,
        expected_sha256,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        temporary_pdf = (
            Path(temp_dir) / "tampered_cv.pdf"
        )

        temporary_pdf.write_bytes(
            source_pdf.read_bytes()
        )

        copy_before = verify_document(
            temporary_pdf,
            expected_sha256,
        )

        # Append bytes to a temporary copy only.
        # This deliberately changes its fingerprint.
        with temporary_pdf.open("ab") as file:
            file.write(
                b"\n% Synthetic tamper test\n"
            )

        copy_after = verify_document(
            temporary_pdf,
            expected_sha256,
        )

    original_after = verify_document(
        source_pdf,
        expected_sha256,
    )

    passed = (
        original_check["valid"]
        and copy_before["valid"]
        and not copy_after["valid"]
        and copy_after["reason"]
            == "document_fingerprint_mismatch"
        and original_after["valid"]
    )

    return {
        "test_name": (
            "temporary_pdf_modification_detection"
        ),
        "original_before": original_check,
        "temporary_copy_before": copy_before,
        "temporary_copy_after": copy_after,
        "original_after": original_after,
        "passed": passed,
    }


def save_json(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main():
    print(
        "=== LESSON 6C.1: "
        "DOCUMENT FINGERPRINTS ==="
    )

    text_fingerprint = fingerprint_document(
        TEXT_PDF
    )

    scanned_fingerprint = fingerprint_document(
        SCANNED_PDF
    )

    print("\nTEXT PDF")
    print(
        "File:",
        text_fingerprint["source_file"],
    )
    print(
        "SHA-256:",
        text_fingerprint["sha256"],
    )
    print(
        "Size:",
        text_fingerprint["size_bytes"],
        "bytes",
    )

    print("\nSCANNED PDF")
    print(
        "File:",
        scanned_fingerprint["source_file"],
    )
    print(
        "SHA-256:",
        scanned_fingerprint["sha256"],
    )
    print(
        "Size:",
        scanned_fingerprint["size_bytes"],
        "bytes",
    )

    fingerprints_report = {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "algorithm": "sha256",
        "documents": [
            text_fingerprint,
            scanned_fingerprint,
        ],
    }

    save_json(
        FINGERPRINTS_OUTPUT,
        fingerprints_report,
    )

    visual_report = load_visual_report()

    bound_evidence = bind_visual_evidence(
        visual_report,
        scanned_fingerprint,
    )

    save_json(
        BOUND_EVIDENCE_OUTPUT,
        bound_evidence,
    )

    print("\nEVIDENCE BINDING")
    print(
        "Binding type:",
        bound_evidence["binding_type"],
    )
    print(
        "Evidence items:",
        bound_evidence["evidence_count"],
    )
    print(
        "Source SHA-256 matches:",
        bound_evidence["source_sha256"]
            == scanned_fingerprint["sha256"],
    )

    tamper_result = test_tamper_detection(
        SCANNED_PDF,
        scanned_fingerprint["sha256"],
    )

    save_json(
        TAMPER_TEST_OUTPUT,
        tamper_result,
    )

    print("\nTAMPER DETECTION")
    print(
        "Original matches:",
        tamper_result[
            "original_before"
        ]["valid"],
    )
    print(
        "Temporary copy matches:",
        tamper_result[
            "temporary_copy_before"
        ]["valid"],
    )
    print(
        "Modified copy accepted:",
        tamper_result[
            "temporary_copy_after"
        ]["valid"],
    )
    print(
        "Original still matches:",
        tamper_result[
            "original_after"
        ]["valid"],
    )
    print(
        "Tamper test passed:",
        tamper_result["passed"],
    )

    print("\nSaved:", FINGERPRINTS_OUTPUT)
    print("Saved:", BOUND_EVIDENCE_OUTPUT)
    print("Saved:", TAMPER_TEST_OUTPUT)

    if not tamper_result["passed"]:
        raise SystemExit(
            "Tamper detection test failed."
        )

    if bound_evidence["evidence_count"] != 6:
        raise SystemExit(
            "Unexpected visual evidence count."
        )

    print(
        "\n=== LESSON 6C.1 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
