
"""Lesson 5G.1: Audit display excerpts for skill evidence."""

from scripts.demo_parent_cv_verification import (
    load_chunks,
    collect_candidate_chunks,
    find_skill_evidence,
)

from scripts.privacy_safe_evidence import (
    make_display_excerpt,
    EMAIL_PATTERN,
    PHONE_PATTERN,
)


CANDIDATES = [
    "r00001",
    "r00041",
    "r00038",
    "r00049",
    "r00053",
]

SKILLS = [
    "Python",
    "SQL",
]


def main():
    chunks = load_chunks()

    parents = collect_candidate_chunks(
        chunks
    )

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    total = 0
    missing = 0
    leaked_contacts = 0

    print(
        "=== LESSON 5G.1: "
        "PRIVACY AUDIT ==="
    )

    for candidate_id in CANDIDATES:
        print(
            f"\nCandidate {candidate_id}"
        )

        matches = find_skill_evidence(
            parents[candidate_id],
            SKILLS,
        )

        for skill in SKILLS:
            chunk_ids = matches.get(
                skill,
                [],
            )

            if not chunk_ids:
                missing += 1
                print(
                    f"  {skill}: NO SOURCE"
                )
                continue

            chunk_id = chunk_ids[0]

            excerpt = make_display_excerpt(
                chunk_lookup[chunk_id]["text"],
                skill,
            )

            if excerpt is None:
                missing += 1
                print(
                    f"  {skill}: NO EXCERPT"
                )
                continue

            total += 1

            has_email = bool(
                EMAIL_PATTERN.search(excerpt)
            )

            has_phone = bool(
                PHONE_PATTERN.search(excerpt)
            )

            if has_email or has_phone:
                leaked_contacts += 1

            print(
                f"  {skill} [{chunk_id}]"
            )

            print(
                f"    {excerpt}"
            )

            print(
                "    Contact-pattern check:",
                (
                    "WARNING"
                    if has_email or has_phone
                    else "PASS"
                ),
            )

    print(
        "\n=== AUDIT SUMMARY ==="
    )

    print(
        "Excerpts generated:",
        total,
    )

    print(
        "Missing evidence:",
        missing,
    )

    print(
        "Contact-pattern warnings:",
        leaked_contacts,
    )

    if (
        total == len(CANDIDATES) * len(SKILLS)
        and missing == 0
        and leaked_contacts == 0
    ):
        print(
            "Basic privacy audit: PASS"
        )
    else:
        print(
            "Basic privacy audit: NEEDS REVIEW"
        )


if __name__ == "__main__":
    main()
