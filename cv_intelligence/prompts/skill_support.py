
"""Conservative skill support checks for LLM proposals."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .response_parser import PromptResponseError
from .injection_guard import reject_instruction_like_evidence


def _explicit_skill_mention(skill: str, text: str) -> bool:
    """Match an explicit skill name, not a substring."""

    pattern = (
        r"(?<!\w)"
        + re.escape(skill.strip())
        + r"(?!\w)"
    )

    return bool(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
    )


def validate_skill_support(
    *,
    claims: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reject claims without an explicit cited skill mention."""

    # Step 1: Detect suspicious instructions inside CV evidence.
    reject_instruction_like_evidence(evidence=evidence)

    # Step 2: Build the evidence lookup.
    evidence_by_id = {}

    for item in evidence:
        evidence_id = item["evidence_id"]

        if evidence_id in evidence_by_id:
            raise PromptResponseError(
                "Duplicate evidence ID"
            )

        evidence_by_id[evidence_id] = item

    # Step 3: Validate each proposed skill claim.
    validated = []

    for claim in claims:
        skill = claim["skill"].strip()
        candidate_id = claim["candidate_id"]

        supported = False

        for evidence_id in claim["evidence_ids"]:
            item = evidence_by_id.get(evidence_id)

            if item is None:
                raise PromptResponseError(
                    "Unknown evidence ID"
                )

            if item["candidate_id"] != candidate_id:
                raise PromptResponseError(
                    "Evidence ownership mismatch"
                )

            if _explicit_skill_mention(
                skill,
                item["text"],
            ):
                supported = True

        if not supported:
            raise PromptResponseError(
                "Claim lacks explicit skill evidence"
            )

        validated.append(dict(claim))

    return validated
