
"""Build LLM prompts from already-verified candidate evidence.

This module does not call an LLM or modify eligibility.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .candidate_analysis import build_candidate_messages


def build_verified_candidate_messages(
    *,
    query: str,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Prepare prompts using only accepted evidence."""

    evidence = []
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

            evidence.append({
                "evidence_id": f"verified-{counter}",
                "candidate_id": candidate_id,
                "text": text,
            })

    if not evidence:
        raise ValueError("No verified evidence available")

    return build_candidate_messages(
        query=query,
        evidence=evidence,
    )
