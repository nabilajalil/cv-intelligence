
"""Validate LLM claim proposals against prompt evidence."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .claim_validation import validate_claim_ownership
from .response_parser import parse_claim_proposals


def validate_llm_proposals(
    *,
    raw_response: str,
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Validate response schema, IDs, and candidate ownership."""

    if (
        len(messages) != 2
        or messages[0].get("role") != "system"
        or messages[1].get("role") != "user"
    ):
        raise ValueError("Invalid prompt messages")

    payload = json.loads(messages[1]["content"])
    evidence = payload["evidence"]

    allowed_evidence_ids = {
        item["evidence_id"] for item in evidence
    }

    allowed_candidate_ids = {
        item["candidate_id"] for item in evidence
    }

    claims = parse_claim_proposals(
        raw_response,
        allowed_evidence_ids=allowed_evidence_ids,
        allowed_candidate_ids=allowed_candidate_ids,
    )

    return validate_claim_ownership(
        claims=claims,
        evidence=evidence,
    )
