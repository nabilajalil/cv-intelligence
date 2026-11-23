
"""Lesson 6B.4 — PDF page provenance and evidence validation.

Validation levels:
1. verified_pdf_text:
   Evidence matches text extracted from a PDF page.

2. vision_profile_only:
   Evidence matches the saved vision profile, but has
   NOT been independently verified against the image.

3. unsupported:
   Evidence cannot be matched to the relevant source.

No OpenAI API calls.

Run:
    python -m scripts.multimodal.validate_multimodal_pdf_evidence
"""

import json
import re
from pathlib import Path

import pymupdf

from scripts.multimodal.demo_multimodal_evidence import (
    search_with_evidence,
)
from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
)


OUTPUT_DIR = Path(
    "data/multimodal/retrieval"
)

QUERIES = [
    "Python Docker",
    "Recruitment Analytics Dashboard",
    "Kubernetes Terraform",
]


def normalize(text):
    """Normalize whitespace and punctuation for matching."""

    if not isinstance(text, str):
        return ""

    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            text.casefold(),
        )
    )


def source_contains_evidence(profile, field, evidence):
    """Check whether saved structured data contains evidence."""

    value = profile.get(field)

    if isinstance(value, str):
        entries = [value]

    elif isinstance(value, list):
        entries = [
            entry
            for entry in value
            if isinstance(entry, str)
        ]

    else:
        return False

    normalized_evidence = normalize(evidence)

    return bool(normalized_evidence) and any(
        normalize(entry) == normalized_evidence
        for entry in entries
    )


def find_pdf_text_pages(pdf_path, evidence):
    """Find pages whose extractable text contains evidence."""

    normalized_evidence = normalize(evidence)

    if not normalized_evidence:
        return []

    matching_pages = []

    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            page_text = normalize(
                page.get_text("text")
            )

            if normalized_evidence in page_text:
                matching_pages.append(page_number)

    return matching_pages


def validate_source(evidence, source, record):
    """Validate evidence for one PDF source."""

    pdf_path = Path(
        source["source_file"]
    )

    method = source["extraction_method"]

    result = {
        "source_file": str(pdf_path),
        "extraction_method": method,
        "status": "unsupported",
        "page_numbers": [],
        "reason": "",
    }

    if not pdf_path.is_file():
        result["reason"] = (
            "Source PDF is missing."
        )
        return result

    if not source_contains_evidence(
        record["profile"],
        evidence["field"],
        evidence["text"],
    ):
        result["reason"] = (
            "Evidence is absent from this source profile."
        )
        return result

    try:
        matching_pages = find_pdf_text_pages(
            pdf_path,
            evidence["text"],
        )

        with pymupdf.open(pdf_path) as document:
            page_count = len(document)

    except Exception as error:
        result["reason"] = (
            f"Unable to inspect PDF: {error}"
        )
        return result

    if matching_pages:
        result["status"] = "verified_pdf_text"
        result["page_numbers"] = matching_pages
        result["reason"] = (
            "Evidence matches extractable PDF page text."
        )

    elif method == "vision":
        result["status"] = "vision_profile_only"

        # The existing vision extraction processed
        # only the first page.
        if page_count > 0:
            result["page_numbers"] = [1]

        result["reason"] = (
            "Evidence appears in the saved vision profile. "
            "Page 1 is the processing provenance, "
            "not an independently verified image match."
        )

    else:
        result["reason"] = (
            "Evidence does not match extractable PDF text."
        )

    return result


def main():
    print(
        "=== LESSON 6B.4: "
        "PDF EVIDENCE VALIDATION ==="
    )

    records = load_profiles()

    record_lookup = {
        (
            record["candidate_id"],
            record["source_file"],
            record["extraction_method"],
        ): record
        for record in records
    }

    all_reports = []

    status_counts = {
        "verified_pdf_text": 0,
        "vision_profile_only": 0,
        "unsupported": 0,
    }

    for query in QUERIES:
        print("\nQUERY:", query)

        results = search_with_evidence(
            query,
            records,
        )

        query_report = {
            "query": query,
            "candidates": [],
        }

        for candidate in results:
            candidate_report = {
                "candidate_id": candidate[
                    "candidate_id"
                ],
                "evidence": [],
            }

            print(
                "Candidate:",
                candidate["candidate_id"],
            )

            for evidence in candidate["evidence"]:
                evidence_report = {
                    "field": evidence["field"],
                    "text": evidence["text"],
                    "validations": [],
                }

                print(
                    "\nEvidence:",
                    evidence["text"],
                )

                for source in evidence["sources"]:
                    key = (
                        candidate["candidate_id"],
                        source["source_file"],
                        source["extraction_method"],
                    )

                    record = record_lookup.get(key)

                    if record is None:
                        validation = {
                            "source_file": source[
                                "source_file"
                            ],
                            "extraction_method": source[
                                "extraction_method"
                            ],
                            "status": "unsupported",
                            "page_numbers": [],
                            "reason": (
                                "Source record not found."
                            ),
                        }

                    else:
                        validation = validate_source(
                            evidence,
                            source,
                            record,
                        )

                    status_counts[
                        validation["status"]
                    ] += 1

                    evidence_report[
                        "validations"
                    ].append(validation)

                    print(
                        "  ",
                        validation[
                            "extraction_method"
                        ],
                        "->",
                        validation["status"],
                        "| pages:",
                        validation["page_numbers"],
                    )

                candidate_report[
                    "evidence"
                ].append(evidence_report)

            query_report[
                "candidates"
            ].append(candidate_report)

        if not results:
            print("No matching evidence.")

        all_reports.append(query_report)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "pdf_evidence_validation.json"
    )

    output_path.write_text(
        json.dumps(
            {
                "queries": all_reports,
                "status_counts": status_counts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nVALIDATION SUMMARY")

    for status, count in status_counts.items():
        print(
            f"{status}: {count}"
        )

    print("\nSaved:", output_path)

    print(
        "\n=== VALIDATION COMPLETE ==="
    )


if __name__ == "__main__":
    main()
