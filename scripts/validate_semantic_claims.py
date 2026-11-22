
"""Lesson 5G.3: Deterministic semantic claim validation.

This validator supports a deliberately narrow claim:
a particular skill is explicitly mentioned in a CV.

It does NOT establish:
- Professional experience
- Years of experience
- Proficiency
- Project usage
- Employment history

The statement is rendered by Python from validated fields,
not copied from an LLM.
"""

import argparse
import json
import re
from pathlib import Path


RESULTS_DIR = Path(
    "data/public_benchmark/graph/privacy_safe_answers"
)

SUPPORTED_CLAIM_TYPE = "skill_mentioned"


def skill_pattern(skill):
    return re.compile(
        r"(?<![A-Za-z0-9])"
        + re.escape(skill)
        + r"(?![A-Za-z0-9])",
        flags=re.IGNORECASE,
    )


def validate_claim(claim, registry, selected):
    """Return a validated canonical claim or a rejection.

    Never trust a free-text statement supplied by an LLM.
    """

    candidate_id = claim.get("candidate_id")
    claim_type = claim.get("claim_type")
    skill = claim.get("skill")
    evidence_id = claim.get("evidence_id")

    if candidate_id not in selected:
        return None, "CANDIDATE_NOT_SELECTED"

    if claim_type != SUPPORTED_CLAIM_TYPE:
        return None, "UNSUPPORTED_CLAIM_TYPE"

    if not isinstance(skill, str) or not skill.strip():
        return None, "INVALID_SKILL"

    if not isinstance(evidence_id, str):
        return None, "INVALID_EVIDENCE_ID"

    evidence = registry.get(evidence_id)

    if evidence is None:
        return None, "EVIDENCE_NOT_FOUND"

    if evidence.get("candidate_id") != candidate_id:
        return None, "EVIDENCE_CANDIDATE_MISMATCH"

    if evidence.get("type") != "skill":
        return None, "WRONG_EVIDENCE_TYPE"

    if evidence.get("subject", "").casefold() != skill.casefold():
        return None, "SKILL_MISMATCH"

    excerpt = evidence.get("display_excerpt")

    if not isinstance(excerpt, str):
        return None, "MISSING_DISPLAY_EXCERPT"

    if not skill_pattern(skill).search(excerpt):
        return None, "SKILL_NOT_IN_EXCERPT"

    validated = {
        "candidate_id": candidate_id,
        "claim_type": SUPPORTED_CLAIM_TYPE,
        "skill": evidence["subject"],
        "evidence_id": evidence_id,
        "chunk_id": evidence["chunk_id"],
        "statement": (
            f"The CV for candidate {candidate_id} "
            f"explicitly mentions {evidence['subject']}."
        ),
    }

    return validated, None


def build_test_claims(registry, selected):
    """Create supported and deliberately unsupported cases."""

    skill_records = [
        record
        for record in registry.values()
        if record.get("type") == "skill"
        and record.get("candidate_id") in selected
    ]

    if not skill_records:
        raise ValueError(
            "No selected candidate skill evidence found."
        )

    first = skill_records[0]

    base = {
        "candidate_id": first["candidate_id"],
        "skill": first["subject"],
        "evidence_id": first["evidence_id"],
    }

    tests = [
        {
            "name": "Supported skill mention",
            "claim": {
                **base,
                "claim_type": "skill_mentioned",
            },
            "expect_accepted": True,
        },
        {
            "name": "Unsupported professional experience",
            "claim": {
                **base,
                "claim_type": "professional_experience",
                "statement": (
                    "The candidate has professionally "
                    "used this skill."
                ),
            },
            "expect_accepted": False,
        },
        {
            "name": "Unsupported five years experience",
            "claim": {
                **base,
                "claim_type": "years_experience",
                "years": 5,
            },
            "expect_accepted": False,
        },
        {
            "name": "Unsupported expert proficiency",
            "claim": {
                **base,
                "claim_type": "expert_proficiency",
            },
            "expect_accepted": False,
        },
        {
            "name": "Unsupported project usage",
            "claim": {
                **base,
                "claim_type": "project_usage",
            },
            "expect_accepted": False,
        },
        {
            "name": "Nonexistent evidence ID",
            "claim": {
                **base,
                "claim_type": "skill_mentioned",
                "evidence_id": "ev_does_not_exist",
            },
            "expect_accepted": False,
        },
        {
            "name": "Unselected candidate",
            "claim": {
                **base,
                "claim_type": "skill_mentioned",
                "candidate_id": "r99999",
            },
            "expect_accepted": False,
        },
        {
            "name": "Wrong skill for evidence",
            "claim": {
                **base,
                "claim_type": "skill_mentioned",
                "skill": "Kubernetes",
            },
            "expect_accepted": False,
        },
        {
            "name": "Free-text overclaim with valid type",
            "claim": {
                **base,
                "claim_type": "skill_mentioned",
                "statement": (
                    "This candidate is a senior expert "
                    "with 10 years of professional experience."
                ),
            },
            "expect_accepted": True,
        },
    ]

    return tests


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default=str(
            RESULTS_DIR / "cf54dc156415.json"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Run Lesson 5G.2 first: {input_path}"
        )

    data = json.loads(
        input_path.read_text(
            encoding="utf-8"
        )
    )

    registry = data["safe_evidence_registry"]

    selected = data["selected_candidates"]

    print(
        "=== LESSON 5G.3: "
        "SEMANTIC CLAIM VALIDATION ==="
    )

    print("Selected candidates:", len(selected))
    print("Evidence records:", len(registry))

    tests = build_test_claims(
        registry,
        selected,
    )

    passed = 0
    accepted_count = 0
    rejected_count = 0

    for index, test in enumerate(
        tests,
        start=1,
    ):
        validated, reason = validate_claim(
            test["claim"],
            registry,
            selected,
        )

        accepted = validated is not None

        if accepted:
            accepted_count += 1
        else:
            rejected_count += 1

        correct = (
            accepted == test["expect_accepted"]
        )

        if correct:
            passed += 1

        print(
            f"\nTest {index}: {test['name']}"
        )

        print(
            "  Decision:",
            "ACCEPT" if accepted else "REJECT",
        )

        print(
            "  Reason:",
            reason or "SUPPORTED_CLAIM_TYPE",
        )

        if validated:
            print(
                "  Canonical statement:",
                validated["statement"],
            )

        print(
            "  Test:",
            "PASS" if correct else "FAIL",
        )

    print(
        "\n=== VALIDATION SUMMARY ==="
    )

    print("Tests:", len(tests))
    print("Passed:", passed)
    print("Accepted:", accepted_count)
    print("Rejected:", rejected_count)

    if passed == len(tests):
        print(
            "Semantic claim validation: PASS"
        )
    else:
        print(
            "Semantic claim validation: NEEDS REVIEW"
        )


if __name__ == "__main__":
    main()
