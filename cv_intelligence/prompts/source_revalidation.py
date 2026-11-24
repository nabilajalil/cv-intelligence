
"""Revalidate mapped evidence against the original PDF."""

from collections.abc import Mapping
from typing import Any

from scripts.multimodal.hash_enforced_evidence import (
    validate_hash_enforced_claim,
)
from scripts.multimodal.multimodal_claim_safety_gate import (
    make_claim,
)

from .response_parser import PromptResponseError


def revalidate_mapped_evidence(
    *,
    evidence_id: str,
    candidate_id: str,
    evidence_mapping: Mapping[str, Mapping[str, Any]],
    source_profile: Mapping[str, Any],
    bound_report: Mapping[str, Any],
    pdf_path=None,
) -> dict[str, Any]:
    """Return evidence only if original-source checks pass."""

    item = evidence_mapping.get(evidence_id)

    if item is None:
        raise PromptResponseError("Unknown evidence ID")

    if item.get("candidate_id") != candidate_id:
        raise PromptResponseError(
            "Evidence ownership mismatch"
        )

    claim = make_claim(
        candidate_id=candidate_id,
        field=item["field"],
        text=item["text"],
        source_file=item["source_file"],
        page_number=item["page_number"],
        claim_type="cv_mention",
    )

    decision = validate_hash_enforced_claim(
        claim,
        source_profile,
        bound_report,
        pdf_path=pdf_path,
    )

    if (
        decision.get("decision") != "accept"
        or decision.get("integrity_status") != "match"
    ):
        raise PromptResponseError(
            "Original-source evidence validation failed"
        )

    return {
        "evidence_id": evidence_id,
        "candidate_id": candidate_id,
        "field": item["field"],
        "text": item["text"],
        "source_file": item["source_file"],
        "page_number": item["page_number"],
    }
