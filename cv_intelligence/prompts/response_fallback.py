"""Deterministic fallback for recruiter response presentation."""

from .response_context import build_response_context
from .response_renderer import render_response
from .response_validation import PresentationChoice


MAX_SECTION_LABEL_LENGTH = 100
MAX_SECTIONS_PER_SKILL = 5


def build_fallback_response(candidate: dict) -> str:
    """Render evaluated facts without requiring LLM inference."""

    if candidate["required_skill_count"] == 0:
        return (
            "No explicit technical skills were requested. "
            "Skill matching was not performed."
        )

    context = build_response_context(candidate)

    safe_sections = {}

    for skill, sections in context["supporting_sections"].items():
        safe_sections[skill] = [
            section
            for section in sections
            if isinstance(section, str)
            and section.strip()
            and len(section) <= MAX_SECTION_LABEL_LENGTH
            and section.isprintable()
        ][:MAX_SECTIONS_PER_SKILL]

    context["supporting_sections"] = safe_sections

    return render_response(
        context,
        PresentationChoice(style="professional"),
    )
