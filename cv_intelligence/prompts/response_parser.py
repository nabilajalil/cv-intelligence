
"""Strict parser for untrusted LLM claim proposals."""

import json
from typing import Any


class PromptResponseError(ValueError):
    """Raised when an LLM response violates the expected schema."""


def parse_claim_proposals(
    raw_response: str,
    *,
    allowed_evidence_ids: set[str],
    allowed_candidate_ids: set[str],
) -> list[dict[str, Any]]:
    """Parse proposed claims; do not treat them as verified."""

    if not isinstance(raw_response, str):
        raise PromptResponseError("Response must be text")

    if len(raw_response) > 20000:
        raise PromptResponseError("Response too large")

    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise PromptResponseError("Invalid JSON") from exc

    if not isinstance(data, dict) or set(data) != {"claims"}:
        raise PromptResponseError("Unexpected response structure")

    claims = data["claims"]

    if not isinstance(claims, list) or len(claims) > 50:
        raise PromptResponseError("Invalid claims list")

    validated = []

    for claim in claims:
        if not isinstance(claim, dict):
            raise PromptResponseError("Invalid claim")

        if set(claim) != {
            "candidate_id",
            "skill",
            "evidence_ids",
        }:
            raise PromptResponseError("Unexpected claim fields")

        candidate_id = claim["candidate_id"]
        skill = claim["skill"]
        evidence_ids = claim["evidence_ids"]

        if (
            not isinstance(candidate_id, str)
            or candidate_id not in allowed_candidate_ids
        ):
            raise PromptResponseError("Unknown candidate")

        if (
            not isinstance(skill, str)
            or not skill.strip()
            or len(skill) > 100
        ):
            raise PromptResponseError("Invalid skill")

        if (
            not isinstance(evidence_ids, list)
            or not evidence_ids
            or len(evidence_ids) > 20
            or any(
                not isinstance(eid, str)
                or eid not in allowed_evidence_ids
                for eid in evidence_ids
            )
        ):
            raise PromptResponseError("Unknown evidence reference")

        validated.append(
            {
                "candidate_id": candidate_id,
                "skill": skill.strip(),
                "evidence_ids": evidence_ids,
            }
        )

    return validated
