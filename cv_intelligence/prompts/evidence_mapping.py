
"""Map prompt-local IDs to original protected evidence records."""

from collections.abc import Mapping, Sequence
from typing import Any


def build_evidence_mapping(
    *,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build a prompt ID mapping from protected evaluations.

    Caller must supply evaluations produced by the trusted
    protected retrieval and verification pipeline.
    """

    mapping = {}
    counter = 0

    for candidate in candidates:
        if candidate.get("eligible") is not True:
            continue

        candidate_id = candidate.get("candidate_id")

        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("Invalid candidate ID")

        for item in candidate.get("accepted_evidence", []):
            if (
                item.get("decision") != "accept"
                or item.get("integrity_status") != "match"
            ):
                continue

            text = item.get("text")

            if not isinstance(text, str) or not text.strip():
                continue

            counter += 1
            evidence_id = f"verified-{counter}"

            mapping[evidence_id] = {
                "candidate_id": candidate_id,
                "field": item["field"],
                "text": text,
                "source_file": item["source_file"],
                "page_number": item["page_number"],
            }

    if not mapping:
        raise ValueError("No verified evidence available")

    return mapping
