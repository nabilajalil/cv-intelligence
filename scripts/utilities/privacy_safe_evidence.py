
"""Lesson 5G.1B: Privacy-minimized skill evidence.

Prefer extracting the skill-list or technology-list field
rather than redacting a large surrounding text window.

Output is a normalized display excerpt, not an exact
source quotation.

This is a controlled demonstration for the benchmark's
flattened CV format, not a complete PII anonymizer.
"""

import re


SKILL_PATTERN_TEMPLATE = (
    r"(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"
)

FIELD_START_PATTERN = re.compile(
    r"\b(Skills|Tech)\b\s*:?\s*",
    flags=re.IGNORECASE,
)

FIELD_END_PATTERN = re.compile(
    r"\b("
    r"Languages|Preferences|Summary|Education|"
    r"Experience|Employment|Projects|"
    r"Certifications|Achievements"
    r")\b",
    flags=re.IGNORECASE,
)

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@"
    r"[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

PHONE_PATTERN = re.compile(
    r"(?<!\w)\+?\d[\d(). -]{7,}\d(?!\w)"
)

DOB_PATTERN = re.compile(
    r"\b(?:DOB|Date of Birth|Birth Date)"
    r"\s*:?\s*"
    r"(?:\d{1,2}\s+[A-Za-z]+\s+\d{4}|"
    r"\d{4}-\d{2}-\d{2})",
    flags=re.IGNORECASE,
)


def skill_regex(skill):
    return re.compile(
        SKILL_PATTERN_TEMPLATE.format(
            re.escape(skill)
        ),
        flags=re.IGNORECASE,
    )


def find_skill_field(text, skill):
    """Find a Skills/Tech field containing the skill."""

    skill_matcher = skill_regex(skill)

    for field_match in FIELD_START_PATTERN.finditer(
        text
    ):
        start = field_match.end()

        next_field = FIELD_END_PATTERN.search(
            text,
            start,
        )

        end = (
            next_field.start()
            if next_field
            else min(len(text), start + 300)
        )

        # Limit unusually long or malformed fields.
        end = min(end, start + 300)

        content = text[start:end].strip()

        if skill_matcher.search(content):
            label = field_match.group(1)

            return f"{label}: {content}"

    return None


def make_display_excerpt(
    source_text,
    skill,
    context_chars=75,
):
    """Build a minimized excerpt around verified skill text.

    Prefer a structured field; otherwise return only
    the skill token rather than unrelated personal data.
    """

    matcher = skill_regex(skill)

    match = matcher.search(source_text)

    if match is None:
        return None

    field = find_skill_field(
        source_text,
        skill,
    )

    if field is not None:
        excerpt = field
    else:
        excerpt = (
            "Technology mentioned: "
            + match.group(0)
        )

    excerpt = re.sub(
        r"\s+",
        " ",
        excerpt,
    ).strip()

    return excerpt


def test_redaction():
    sample = (
        "Alex Morgan alex@example.com "
        "+1 555-0152 Tashkent, Uzbekistan "
        "DOB: 15 Nov 1998 "
        "Skills Python Docker SQL "
        "Languages English"
    )

    excerpt = make_display_excerpt(
        sample,
        "Python",
    )

    assert excerpt is not None
    assert "Python" in excerpt
    assert "alex@example.com" not in excerpt
    assert "555-0152" not in excerpt
    assert "DOB" not in excerpt
    assert "Tashkent" not in excerpt

    # Regression test: "Tech" must not match "Technical".
    technical_sample = (
        "Lead Technical Writer "
        "writer@example.com "
        "+1 555-0167 "
        "Leipzig, Germany "
        "DOB: 19 Sep 1989 "
        "Skills Communication Confluence "
        "Gitlab Sql Agile Scrum Linux "
        "Languages English"
    )

    sql_excerpt = make_display_excerpt(
        technical_sample,
        "SQL",
    )

    assert sql_excerpt is not None
    assert sql_excerpt.startswith("Skills:")
    assert "Sql" in sql_excerpt

    assert "Technical Writer" not in sql_excerpt
    assert "writer@example.com" not in sql_excerpt
    assert "555-0167" not in sql_excerpt
    assert "Leipzig" not in sql_excerpt
    assert "DOB" not in sql_excerpt

    print("Privacy minimization smoke test: PASS")
    print("Technical/Tech regression test: PASS")



if __name__ == "__main__":
    test_redaction()
