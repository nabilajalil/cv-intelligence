
"""Lesson 6B.8 — Evidence-grounded multimodal answers.

Combines:
1. Multimodal evidence retrieval.
2. Source-specific visual verification.
3. Deterministic claim safety checks.
4. Evidence-grounded answer rendering.

No OpenAI API calls.

Run:
    python -m scripts.demo_grounded_multimodal_answers
"""

import json
import re
from pathlib import Path

from scripts.demo_multimodal_retrieval import (
    load_profiles,
)

from scripts.demo_multimodal_evidence import (
    search_with_evidence,
)

from scripts.multimodal_claim_safety_gate import (
    PROFILE_PATH,
    VISUAL_REPORT_PATH,
    build_verification_lookup,
    load_json,
    make_claim,
    validate_claim,
)


OUTPUT_PATH = Path(
    "data/multimodal/reports/"
    "grounded_multimodal_answers.json"
)

QUERIES = [
    "Python Docker",
    "Recruitment Analytics Dashboard",
    "Kubernetes Terraform",
]

# These are the explicit skill requirements used in
# our controlled demonstration.
REQUIRED_SKILLS = {
    "Python Docker": ["Python", "Docker"],
    "Kubernetes Terraform": [
        "Kubernetes",
        "Terraform",
    ],
}


def normalize(text):
    """Normalize words for exact skill comparison."""

    if not isinstance(text, str):
        return ""

    return " ".join(
        re.findall(
            r"[a-z0-9]+",
            text.casefold(),
        )
    )


def source_matches(record_source, expected_source):
    """Compare source paths without relying on spelling."""

    return (
        Path(record_source).resolve()
        == Path(expected_source).resolve()
    )


def build_source_context():
    """Load the scanned source and its visual decisions."""

    source_profile = load_json(
        PROFILE_PATH
    )

    visual_report = load_json(
        VISUAL_REPORT_PATH
    )

    if not source_matches(
        source_profile["source_file"],
        visual_report["source_file"],
    ):
        raise ValueError(
            "Visual report source does not match "
            "the scanned profile."
        )

    if not source_matches(
        source_profile["source_file"],
        visual_report["source_file"],
    ):
        raise ValueError(
            "Source provenance mismatch."
        )

    # Keep the original path spelling used by the
    # safety gate's exact source comparison.
    if (
        source_profile["source_file"]
        != visual_report["source_file"]
    ):
        raise ValueError(
            "Source path strings differ."
        )

    return {
        "source_profile": source_profile,
        "verification_lookup": (
            build_verification_lookup(
                visual_report
            )
        ),
    }


def find_source_evidence(evidence, source_file):
    """Return True if this evidence cites the source."""

    return any(
        source_matches(
            source["source_file"],
            source_file,
        )
        for source in evidence.get("sources", [])
    )


def make_evidence_claim(
    candidate_id,
    evidence,
    source_profile,
):
    """Create a claim for the scanned PDF source."""

    return make_claim(
        candidate_id=candidate_id,
        field=evidence["field"],
        text=evidence["text"],
        source_file=source_profile["source_file"],
        page_number=1,
        claim_type="cv_mention",
    )


def evaluate_candidate(
    candidate,
    context,
    required_skills,
):
    """Apply the safety gate to retrieved evidence."""

    source_profile = context[
        "source_profile"
    ]

    lookup = context[
        "verification_lookup"
    ]

    accepted = []
    withheld = []

    for evidence in candidate["evidence"]:
        if not find_source_evidence(
            evidence,
            source_profile["source_file"],
        ):
            continue

        claim = make_evidence_claim(
            candidate["candidate_id"],
            evidence,
            source_profile,
        )

        decision = validate_claim(
            claim,
            source_profile,
            lookup,
        )

        item = {
            "field": evidence["field"],
            "text": evidence["text"],
            "source_file": claim["source_file"],
            "page_number": claim["page_number"],
            "decision": decision["decision"],
            "reason": decision["reason"],
        }

        if decision["decision"] == "accept":
            accepted.append(item)
        else:
            withheld.append(item)

    verified_skills = {
        normalize(item["text"])
        for item in accepted
        if item["field"] == "skills"
    }

    missing_skills = [
        skill
        for skill in required_skills
        if normalize(skill) not in verified_skills
    ]

    eligible = (
        bool(accepted)
        and not missing_skills
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "eligible": eligible,
        "required_skills": required_skills,
        "missing_verified_skills": missing_skills,
        "accepted_evidence": accepted,
        "withheld_evidence": withheld,
    }


def render_answer(query, evaluations):
    """Render only accepted claims as CV statements."""

    eligible = [
        item
        for item in evaluations
        if item["eligible"]
    ]

    if not eligible:
        return (
            f"Query: {query}\n"
            "No candidates have sufficient "
            "verified evidence for this query."
        )

    lines = [
        f"Query: {query}",
        "",
        "Candidates with verified CV mentions:",
    ]

    for candidate in eligible:
        lines.append(
            f"\nCandidate: {candidate['candidate_id']}"
        )

        for evidence in candidate[
            "accepted_evidence"
        ]:
            lines.append(
                "- CV states "
                f"({evidence['field']}): "
                f"{evidence['text']}"
            )

            lines.append(
                "  Source: "
                f"{evidence['source_file']}"
                f", page {evidence['page_number']}"
            )

        withheld_count = len(
            candidate["withheld_evidence"]
        )

        if withheld_count:
            lines.append(
                f"  Withheld evidence items: "
                f"{withheld_count}"
            )

    lines.append(
        "\nNote: These are verified CV mentions, "
        "not independently confirmed qualifications "
        "or professional expertise."
    )

    return "\n".join(lines)


def main():
    print(
        "=== LESSON 6B.8: "
        "EVIDENCE-GROUNDED MULTIMODAL ANSWERS ==="
    )

    records = load_profiles()
    context = build_source_context()

    print(
        "Loaded profile records:",
        len(records),
    )

    print(
        "Verified source:",
        context["source_profile"]["source_file"],
    )

    results = []

    for query in QUERIES:
        print("\n" + "=" * 60)
        print("QUERY:", query)
        print("=" * 60)

        candidates = search_with_evidence(
            query,
            records,
        )

        required_skills = (
            REQUIRED_SKILLS.get(query, [])
        )

        evaluations = [
            evaluate_candidate(
                candidate,
                context,
                required_skills,
            )
            for candidate in candidates
        ]

        answer = render_answer(
            query,
            evaluations,
        )

        print(answer)

        results.append({
            "query": query,
            "required_skills": required_skills,
            "evaluations": evaluations,
            "answer": answer,
        })

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT_PATH)

    print(
        "\n=== LESSON 6B.8 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
