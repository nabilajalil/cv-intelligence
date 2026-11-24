
"""Validate LLM claim proposals against prompt evidence ownership."""

from collections.abc import Mapping, Sequence
from typing import Any

from .response_parser import PromptResponseError


def validate_claim_ownership(
    *,
    claims: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reject claims referencing another candidate's evidence."""

    ownership = {}

    for item in evidence:
        evidence_id = item["evidence_id"]
        candidate_id = item["candidate_id"]

        if evidence_id in ownership:
            raise PromptResponseError(
                "Duplicate evidence ID"
            )

        ownership[evidence_id] = candidate_id

    validated = []

    for claim in claims:
        candidate_id = claim["candidate_id"]

        for evidence_id in claim["evidence_ids"]:
            if ownership.get(evidence_id) != candidate_id:
                raise PromptResponseError(
                    "Evidence ownership mismatch"
                )

        validated.append(dict(claim))

    return validated
