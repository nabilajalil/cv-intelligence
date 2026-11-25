"""Build compact context from evaluated candidate facts."""


def build_response_context(candidate: dict) -> dict:
    """Prepare candidate facts without exposing raw CV text."""

    matched = candidate["matched_skills"]
    not_found = candidate["skills_not_found_in_evidence"]
    evidence = candidate["skill_evidence"]

    supporting_sections = {}

    for skill in matched:
        supporting_sections[skill] = sorted({
            chunk["section"]
            for chunk in evidence.get(skill, [])
            if chunk.get("section")
        })

    return {
        "matched_skills": list(matched),
        "skills_not_found_in_evidence": list(not_found),
        "matched_skill_count": candidate["matched_skill_count"],
        "required_skill_count": candidate["required_skill_count"],
        "supporting_sections": supporting_sections,
        "evidence_limitation": (
            "Documented skill mentions do not establish "
            "independently verified proficiency or experience."
        ),
    }
