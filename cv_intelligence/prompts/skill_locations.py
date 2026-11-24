
"""Report the CV sections supporting validated skill claims."""

from collections.abc import Mapping, Sequence
from typing import Any


def build_skill_locations(
    *,
    validated_claims: Sequence[Mapping[str, Any]],
    evidence_mapping: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Organize validated evidence by CV section.

    Only call this after validate_secure_llm_response()
    succeeds, using its trusted evidence mapping.
    """

    results = []

    for claim in validated_claims:
        candidate_id = claim["candidate_id"]
        skill = claim["skill"]
        locations = []
        seen_ids = set()

        for evidence_id in claim["evidence_ids"]:
            if evidence_id in seen_ids:
                continue

            seen_ids.add(evidence_id)

            evidence = evidence_mapping[evidence_id]

            if evidence["candidate_id"] != candidate_id:
                raise ValueError("Evidence ownership mismatch")

            locations.append({
                "section": evidence["field"],
                "text": evidence["text"],
                "source_file": evidence["source_file"],
                "page_number": evidence["page_number"],
                "evidence_id": evidence_id,
            })

        results.append({
            "candidate_id": candidate_id,
            "skill": skill,
            "locations": locations,
        })

    return results


def find_all_skill_mentions(
    *,
    candidate_id: str,
    skill: str,
    evidence_mapping: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Find explicit skill mentions in available verified evidence.

    The caller must supply an evidence mapping produced by
    the trusted protected pipeline.
    """

    from .skill_support import _explicit_skill_mention

    if not candidate_id.strip() or not skill.strip():
        raise ValueError("Candidate ID and skill are required")

    locations = []

    for evidence_id, evidence in evidence_mapping.items():
        if evidence["candidate_id"] != candidate_id:
            continue

        if not _explicit_skill_mention(skill, evidence["text"]):
            continue

        locations.append({
            "section": evidence["field"],
            "text": evidence["text"],
            "source_file": evidence["source_file"],
            "page_number": evidence["page_number"],
            "evidence_id": evidence_id,
        })

    return locations


def find_revalidated_skill_mentions(
    *,
    candidate_id: str,
    skill: str,
    evidence_mapping: Mapping[str, Mapping[str, Any]],
    source_profile: Mapping[str, Any],
    bound_report: Mapping[str, Any],
    pdf_path=None,
) -> list[dict[str, Any]]:
    """Return matching mentions that pass original-source validation."""

    from .source_revalidation import revalidate_mapped_evidence

    mentions = find_all_skill_mentions(
        candidate_id=candidate_id,
        skill=skill,
        evidence_mapping=evidence_mapping,
    )

    verified_locations = []

    for mention in mentions:
        evidence_id = mention["evidence_id"]

        # Recheck evidence ownership, source binding and PDF integrity.
        revalidate_mapped_evidence(
            evidence_id=evidence_id,
            candidate_id=candidate_id,
            evidence_mapping=evidence_mapping,
            source_profile=source_profile,
            bound_report=bound_report,
            pdf_path=pdf_path,
        )

        verified_locations.append(mention)

    return verified_locations

