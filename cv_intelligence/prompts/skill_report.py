
"""Format revalidated CV skill locations for human readers."""

from collections.abc import Mapping, Sequence
from typing import Any


def format_skill_report(
    *,
    candidate_id: str,
    skill_locations: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    """Format already-revalidated skill locations as readable text."""

    lines = [
        f"Candidate {candidate_id}",
        "─" * 40,
    ]

    for skill, locations in skill_locations.items():
        lines.append("")
        lines.append(skill)

        if not locations:
            lines.append("  No verified mentions found.")
            continue

        for location in locations:
            section = str(location["section"]).replace("_", " ").title()
            page = location["page_number"]
            evidence = location["text"]

            lines.append(f"  ✓ {section} (Page {page})")
            lines.append(f"    {evidence}")
            lines.append("")

    return "\n".join(lines).rstrip()
