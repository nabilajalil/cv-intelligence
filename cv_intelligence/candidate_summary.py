
def summarize_candidate(candidate: dict) -> str:
    """Create a factual summary from verified skill evidence."""

    matched = candidate["matched_skills"]
    not_found = candidate["skills_not_found_in_evidence"]
    evidence = candidate["skill_evidence"]

    sentences = []

    if matched:
        sentences.append(
            "The candidate's indexed CV contains explicit "
            f"mentions of: {', '.join(matched)}."
        )
    else:
        sentences.append(
            "No requested technical skills were confirmed "
            "in the available indexed CV evidence."
        )

    sections = set()

    for skill in matched:
        for chunk in evidence.get(skill, []):
            section = chunk.get("section")
            if section:
                sections.add(section)

    if sections:
        sentences.append(
            "Supporting mentions appear in the following "
            f"CV sections: {', '.join(sorted(sections))}."
        )

    if not_found:
        sentences.append(
            "No explicit mention was found in the indexed CV "
            f"for: {', '.join(not_found)}."
        )

    sentences.append(
        "These findings indicate documented skill mentions, "
        "not independently verified proficiency or experience."
    )

    return " ".join(sentences)

