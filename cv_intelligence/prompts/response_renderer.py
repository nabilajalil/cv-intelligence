"""Render recruiter responses from evaluated candidate facts."""

from .response_validation import PresentationChoice


def render_response(
    context: dict,
    choice: PresentationChoice,
) -> str:
    """Render an approved style without LLM-generated claims."""

    matched = context["matched_skills"]
    missing = context["skills_not_found_in_evidence"]
    matched_count = context["matched_skill_count"]
    required_count = context["required_skill_count"]
    sections = context["supporting_sections"]
    limitation = context["evidence_limitation"]

    count_sentence = (
        f"Explicit mentions were found for "
        f"{matched_count} of {required_count} requested skills."
        if required_count
        else "No explicit technical skills were requested."
    )

    if matched:
        matched_sentence = (
            "The candidate's indexed CV contains explicit "
            f"mentions of: {', '.join(matched)}."
        )
    else:
        matched_sentence = (
            "No requested technical skill mentions were confirmed."
            if required_count
            else ""
        )

    evidence_sentences = []

    for skill in matched:
        skill_sections = sections.get(skill, [])

        if skill_sections:
            evidence_sentences.append(
                f"Supporting mentions of {skill} appear in: "
                f"{', '.join(skill_sections)}."
            )

    missing_sentences = [
        f"No explicit mention of {skill} was found "
        "in the indexed CV evidence."
        for skill in missing
    ]

    if choice.style == "professional":
        sentences = [
            matched_sentence,
            count_sentence,
            *evidence_sentences,
            *missing_sentences,
        ]

    elif choice.style == "concise":
        sentences = [
            count_sentence,
            f"Documented skill mentions: {', '.join(matched)}."
            if matched else "",
            *evidence_sentences,
            *missing_sentences,
        ]

    else:
        sentences = [
            *evidence_sentences,
            matched_sentence,
            count_sentence,
            *missing_sentences,
        ]

    sentences.append(limitation)

    return " ".join(
        sentence for sentence in sentences if sentence
    )
