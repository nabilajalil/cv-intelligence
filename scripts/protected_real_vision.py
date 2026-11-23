
"""Lesson 6C.4 — Protected real-vision adapter.

Default execution is a FREE dry-run.

The live mode:
- Reads the exact PDF bytes used for hashing.
- Renders the first page from those bytes.
- Calls the existing GPT-4.1-mini verifier.
- Validates every returned evidence decision.
- Checks the source PDF again before publication.
- Saves only a successful, integrity-checked result.

Run free dry-run:
    python -m scripts.protected_real_vision

Live API execution (later):
    python -m scripts.protected_real_vision --live
"""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

from scripts.multimodal.verify_scanned_cv_visual_evidence import (
    PDF_PATH,
    MODEL,
    collect_evidence,
    verify_with_vision,
)


OUTPUT_PATH = Path(
    "data/multimodal/provenance/"
    "protected_real_vision_verification.json"
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_source_snapshot(path):
    """Read one byte snapshot and fingerprint it."""

    path = Path(path)

    if not path.is_file():
        raise RuntimeError("source_file_missing")

    pdf_bytes = path.read_bytes()

    if not pdf_bytes:
        raise RuntimeError("source_file_empty")

    return pdf_bytes, sha256_bytes(pdf_bytes)


def render_snapshot(pdf_bytes):
    """Render the PDF snapshot, not the live file path."""

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf",
    ) as document:

        if len(document) == 0:
            raise RuntimeError("pdf_has_no_pages")

        pixmap = document[0].get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False,
        )

        return pixmap.tobytes("png")


def check_source_unchanged(path, expected_hash):
    """Fail if the current source bytes changed."""

    path = Path(path)

    if not path.is_file():
        raise RuntimeError("source_file_missing")

    actual_hash = sha256_bytes(
        path.read_bytes()
    )

    if actual_hash != expected_hash:
        raise RuntimeError(
            "document_fingerprint_mismatch"
        )


def validate_decisions(evidence, decisions):
    """Require exactly one valid decision per evidence ID."""

    expected_ids = {
        item["evidence_id"]
        for item in evidence
    }

    returned_ids = [
        decision.evidence_id
        for decision in decisions
    ]

    if (
        len(returned_ids) != len(expected_ids)
        or len(set(returned_ids)) != len(returned_ids)
        or set(returned_ids) != expected_ids
    ):
        raise RuntimeError(
            "verification_decision_id_mismatch"
        )

    allowed_statuses = {
        "supported",
        "unsupported",
        "uncertain",
    }

    decisions_by_id = {
        decision.evidence_id: decision
        for decision in decisions
    }

    results = []

    for item in evidence:
        decision = decisions_by_id[
            item["evidence_id"]
        ]

        if decision.status not in allowed_statuses:
            raise RuntimeError(
                "invalid_verification_status"
            )

        results.append({
            **item,
            "status": decision.status,
            "explanation": decision.explanation,
            "source_file": str(PDF_PATH),
            "page_number": 1,
        })

    return results


def publish_record(record, path, expected_hash):
    """Atomically publish after another source check."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=OUTPUT_PATH.parent,
            prefix=".protected_vision_",
            suffix=".tmp",
            delete=False,
        ) as file:

            temporary_path = Path(file.name)

            json.dump(
                record,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.flush()
            os.fsync(file.fileno())

        check_source_unchanged(
            path,
            expected_hash,
        )

        os.replace(
            temporary_path,
            OUTPUT_PATH,
        )

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def main(live=False):
    print(
        "=== LESSON 6C.4: "
        "PROTECTED REAL VISION ==="
    )

    pdf_bytes, source_hash = (
        read_source_snapshot(PDF_PATH)
    )

    evidence = collect_evidence()

    if not evidence:
        raise RuntimeError(
            "no_evidence_to_verify"
        )

    image_bytes = render_snapshot(
        pdf_bytes
    )

    print("Source:", PDF_PATH)
    print("Model:", MODEL)
    print("SHA-256:", source_hash)
    print("PDF bytes:", len(pdf_bytes))
    print("Rendered PNG bytes:", len(image_bytes))
    print("Evidence items:", len(evidence))

    if not live:
        check_source_unchanged(
            PDF_PATH,
            source_hash,
        )

        print("\nDRY-RUN CHECKS")
        print("PDF snapshot: OK")
        print("Page rendering: OK")
        print("Evidence collection: OK")
        print("Source fingerprint: MATCH")
        print("OpenAI API calls: 0")
        print("Verification record saved: NO")
        print("\n=== DRY-RUN PASSED ===")
        return

    print(
        "\nSending one GPT-4.1-mini "
        "verification request..."
    )

    load_dotenv()
    client = OpenAI()

    verification = verify_with_vision(
        client,
        image_bytes,
        evidence,
    )

    results = validate_decisions(
        evidence,
        verification.decisions,
    )

    check_source_unchanged(
        PDF_PATH,
        source_hash,
    )

    counts = {
        status: sum(
            item["status"] == status
            for item in results
        )
        for status in (
            "supported",
            "unsupported",
            "uncertain",
        )
    }

    record = {
        "source_file": str(PDF_PATH),
        "page_number": 1,
        "model": MODEL,
        "verification_type": (
            "pre_post_fingerprint_protected"
        ),
        "hash_algorithm": "sha256",
        "source_sha256": source_hash,
        "verified_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "counts": counts,
        "results": results,
    }

    publish_record(
        record,
        PDF_PATH,
        source_hash,
    )

    print("\nVERIFICATION RESULTS")
    print("Supported:", counts["supported"])
    print("Unsupported:", counts["unsupported"])
    print("Uncertain:", counts["uncertain"])
    print("Source fingerprint: MATCH")
    print("Saved:", OUTPUT_PATH)
    print(
        "\n=== LESSON 6C.4 COMPLETE ==="
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--live",
        action="store_true",
        help=(
            "Make one paid OpenAI API request. "
            "Omit for free dry-run."
        ),
    )

    args = parser.parse_args()

    main(live=args.live)
