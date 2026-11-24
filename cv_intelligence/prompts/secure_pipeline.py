
"""Combine LLM proposal checks with original-source validation.

No LLM calls. This module validates proposed skill claims.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .response_parser import PromptResponseError
from .validation_pipeline import validate_llm_proposals
from .skill_support import validate_skill_support
from .source_revalidation import revalidate_mapped_evidence


def validate_secure_llm_response(
    *,
    raw_response: str,
    messages: Sequence[Mapping[str, str]],
    evidence_mapping: Mapping[str, Mapping[str, Any]],
    source_profile: Mapping[str, Any],
    bound_report: Mapping[str, Any],
    pdf_path=None,
) -> list[dict[str, Any]]:
    """Fail closed unless every proposed claim is verified."""

    # Check that prompt evidence matches the trusted mapping.
    try:
        payload = json.loads(messages[1]["content"])
        prompt_evidence = payload["evidence"]

        if not isinstance(prompt_evidence, list):
            raise ValueError("Invalid prompt evidence")

        if len(prompt_evidence) != len(evidence_mapping):
            raise ValueError("Evidence count mismatch")

        for item in prompt_evidence:
            evidence_id = item["evidence_id"]
            original = evidence_mapping[evidence_id]

            if (
                item["candidate_id"] != original["candidate_id"]
                or item["text"] != original["text"]
            ):
                raise ValueError("Prompt evidence mismatch")

    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise PromptResponseError(
            "Prompt evidence does not match trusted mapping"
        ) from exc

    # Check JSON structure, IDs, and candidate ownership.
    claims = validate_llm_proposals(
        raw_response=raw_response,
        messages=messages,
    )

    # Check that cited evidence explicitly mentions the skill.
    claims = validate_skill_support(
        claims=claims,
        evidence=prompt_evidence,
    )

    # Revalidate every cited source against the original PDF.
    validated = []

    for claim in claims:
        if not claim["evidence_ids"]:
            raise PromptResponseError("Missing evidence")

        # A skill claim must have at least one verified skills
        # record explicitly naming the claimed skill.
        supported_by_skill_field = False

        for evidence_id in claim["evidence_ids"]:
            original = evidence_mapping[evidence_id]

            revalidate_mapped_evidence(
                evidence_id=evidence_id,
                candidate_id=claim["candidate_id"],
                evidence_mapping=evidence_mapping,
                source_profile=source_profile,
                bound_report=bound_report,
                pdf_path=pdf_path,
            )

            if original["field"] == "skills":
                from .skill_support import _explicit_skill_mention

                if _explicit_skill_mention(
                    claim["skill"],
                    original["text"],
                ):
                    supported_by_skill_field = True

        if not supported_by_skill_field:
            raise PromptResponseError(
                "Skill claim lacks verified skills-field evidence"
            )

        validated.append(dict(claim))

    return validated
