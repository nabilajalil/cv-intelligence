
"""Lesson 6C.5 — Protected evidence-grounded answers.

Integrates:
1. Existing multimodal candidate retrieval.
2. Real GPT-4.1-mini verification results.
3. PDF SHA-256 integrity enforcement.
4. Existing evidence-grounded answer rendering.

Includes safe modified/missing PDF tests.

No OpenAI API calls.

Run:
    python -m scripts.demo_protected_grounded_answers
"""

import copy
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
)
from scripts.multimodal.demo_multimodal_evidence import (
    search_with_evidence,
)
from scripts.multimodal.demo_grounded_multimodal_answers import (
    QUERIES,
    REQUIRED_SKILLS,
    find_source_evidence,
    make_evidence_claim,
    normalize,
    render_answer,
)
from scripts.hash_enforced_evidence import (
    validate_hash_enforced_claim,
)
from scripts.multimodal.multimodal_claim_safety_gate import (
    PROFILE_PATH,
    load_json,
)


PROTECTED_REPORT_PATH = Path(
    "data/multimodal/provenance/"
    "protected_real_vision_verification.json"
)

OUTPUT_PATH = Path(
    "data/multimodal/provenance/"
    "protected_grounded_answers.json"
)


def prepare_protected_report(report):
    """Adapt the real protected report to the 6C.2 gate.

    The fingerprint comes from the report-level
    pre/post verification record.

    This is a compatibility adaptation, not a
    new cryptographic verification.
    """

    if (
        report.get("verification_type")
        != "pre_post_fingerprint_protected"
    ):
        raise ValueError(
            "Expected a protected vision report."
        )

    if report.get("hash_algorithm") != "sha256":
        raise ValueError(
            "Expected SHA-256 verification."
        )

    prepared = copy.deepcopy(report)

    for item in prepared["results"]:
        if (
            item.get("source_file")
            != prepared["source_file"]
        ):
            raise ValueError(
                "Evidence source mismatch."
            )

        if (
            item.get("page_number")
            != prepared["page_number"]
        ):
            raise ValueError(
                "Evidence page mismatch."
            )

        if (
            "source_sha256" in item
            and item["source_sha256"]
                != prepared["source_sha256"]
        ):
            raise ValueError(
                "Evidence fingerprint mismatch."
            )

        item["source_sha256"] = (
            prepared["source_sha256"]
        )

    return prepared


def build_protected_context():
    """Load the source profile and protected report."""

    source_profile = load_json(
        PROFILE_PATH
    )

    protected_report = load_json(
        PROTECTED_REPORT_PATH
    )

    if (
        source_profile["source_file"]
        != protected_report["source_file"]
    ):
        raise ValueError(
            "Profile and protected report "
            "refer to different sources."
        )

    return {
        "source_profile": source_profile,
        "protected_report": (
            prepare_protected_report(
                protected_report
            )
        ),
    }


def evaluate_protected_candidate(
    candidate,
    context,
    required_skills,
    pdf_path=None,
):
    """Apply document integrity and claim validation."""

    source_profile = context[
        "source_profile"
    ]

    protected_report = context[
        "protected_report"
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

        decision = validate_hash_enforced_claim(
            claim,
            source_profile,
            protected_report,
            pdf_path=pdf_path,
        )

        item = {
            "field": evidence["field"],
            "text": evidence["text"],
            "source_file": claim["source_file"],
            "page_number": claim["page_number"],
            "decision": decision["decision"],
            "reason": decision["reason"],
            "integrity_status": decision[
                "integrity_status"
            ],
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


def run_query(query, records, context, pdf_path=None):
    """Retrieve candidates and render protected answers."""

    candidates = search_with_evidence(
        query,
        records,
    )

    required_skills = REQUIRED_SKILLS.get(
        query,
        [],
    )

    evaluations = [
        evaluate_protected_candidate(
            candidate,
            context,
            required_skills,
            pdf_path=pdf_path,
        )
        for candidate in candidates
    ]

    answer = render_answer(
        query,
        evaluations,
    )

    return {
        "query": query,
        "evaluations": evaluations,
        "answer": answer,
    }


def main():
    print(
        "=== LESSON 6C.5: "
        "PROTECTED GROUNDED ANSWERS ==="
    )

    records = load_profiles()
    context = build_protected_context()

    print(
        "\nProtected verification:",
        context["protected_report"][
            "verification_type"
        ],
    )

    print(
        "Evidence records:",
        len(
            context["protected_report"][
                "results"
            ]
        ),
    )

    query_results = []

    for query in QUERIES:
        result = run_query(
            query,
            records,
            context,
        )

        query_results.append(result)

        print("\n" + "=" * 50)
        print(result["answer"])

        for evaluation in result["evaluations"]:
            print(
                "Accepted evidence:",
                len(
                    evaluation[
                        "accepted_evidence"
                    ]
                ),
            )

            print(
                "Withheld evidence:",
                len(
                    evaluation[
                        "withheld_evidence"
                    ]
                ),
            )

    print("\n" + "=" * 50)
    print("DOCUMENT INTEGRITY TESTS")

    test_results = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        original_pdf = Path(
            context["source_profile"][
                "source_file"
            ]
        )

        modified_pdf = (
            temp_dir / "modified_cv.pdf"
        )

        modified_pdf.write_bytes(
            original_pdf.read_bytes()
        )

        with modified_pdf.open("ab") as file:
            file.write(
                b"\n% Synthetic tamper test\n"
            )

        modified_result = run_query(
            "Python Docker",
            records,
            context,
            pdf_path=modified_pdf,
        )

        modified_accepted = sum(
            len(item["accepted_evidence"])
            for item in modified_result[
                "evaluations"
            ]
        )

        modified_passed = (
            modified_accepted == 0
        )

        print(
            "Modified PDF accepted evidence:",
            modified_accepted,
        )

        print(
            "Modified PDF test:",
            "PASS" if modified_passed else "FAIL",
        )

        test_results.append({
            "name": "modified_pdf",
            "accepted_evidence": (
                modified_accepted
            ),
            "passed": modified_passed,
        })

        missing_result = run_query(
            "Python Docker",
            records,
            context,
            pdf_path=(
                temp_dir / "missing_cv.pdf"
            ),
        )

        missing_accepted = sum(
            len(item["accepted_evidence"])
            for item in missing_result[
                "evaluations"
            ]
        )

        missing_passed = (
            missing_accepted == 0
        )

        print(
            "Missing PDF accepted evidence:",
            missing_accepted,
        )

        print(
            "Missing PDF test:",
            "PASS" if missing_passed else "FAIL",
        )

        test_results.append({
            "name": "missing_pdf",
            "accepted_evidence": (
                missing_accepted
            ),
            "passed": missing_passed,
        })

    passed = sum(
        item["passed"]
        for item in test_results
    )

    report = {
        "lesson": "6C.5",
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "source_file": context[
            "source_profile"
        ]["source_file"],
        "source_sha256": context[
            "protected_report"
        ]["source_sha256"],
        "queries": query_results,
        "integrity_tests": test_results,
        "integrity_tests_passed": passed,
        "integrity_tests_total": len(
            test_results
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSUMMARY")
    print(
        "Queries evaluated:",
        len(query_results),
    )

    print(
        "Integrity tests passed:",
        f"{passed}/{len(test_results)}",
    )

    print(
        "Saved:",
        OUTPUT_PATH,
    )

    if passed != len(test_results):
        raise SystemExit(
            "Protected grounded-answer tests failed."
        )

    print(
        "\n=== LESSON 6C.5 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
