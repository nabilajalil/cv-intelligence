"""Evaluate constrained recruiter responses using synthetic data."""

import json
from pathlib import Path

from langsmith import Client, tracing_context

from cv_intelligence.prompts.response_service import (
    polish_candidate_response,
)


CASES = [
    ("partial_match", ["Python", "Docker"], ["Kubernetes"]),
    ("full_match", ["Python", "Docker"], []),
    ("no_match", [], ["Python", "Docker"]),
    ("no_explicit_skills", [], []),
]


def build_candidate(name, matched, missing):
    return {
        "candidate_id": f"synthetic-{name}",
        "document_id": f"synthetic-doc-{name}",
        "matched_skills": matched,
        "skills_not_found_in_evidence": missing,
        "matched_skill_count": len(matched),
        "required_skill_count": len(matched) + len(missing),
        "skill_evidence": {
            skill: [{"section": "Projects"}]
            for skill in matched
        },
    }


def main():
    results = []
    client = Client()

    with tracing_context(
        enabled=True,
        project_name="cv-intelligence-evaluation",
    ):
        for name, matched, missing in CASES:
            candidate = build_candidate(name, matched, missing)

            response = polish_candidate_response(candidate)
            summary = response["summary"]

            total = len(matched) + len(missing)

            checks = {
                "matched_skills_preserved": all(
                    skill in summary for skill in matched
                ),
                "missing_evidence_preserved": all(
                    f"No explicit mention of {skill} was found"
                    in summary
                    for skill in missing
                ),
                "counts_preserved": (
                    f"{len(matched)} of {total} requested skills"
                    in summary
                    if total
                    else "Skill matching was not performed" in summary
                ),
                "limitation_preserved": (
                    "independently verified proficiency or experience"
                    in summary
                    if total
                    else True
                ),
                "expected_response_mode": (
                    response["response_mode"] == "llm_assisted"
                    if total
                    else response["fallback_reason"]
                    == "no_explicit_skills"
                ),
            }

            passed = all(checks.values())

            results.append({
                "case": name,
                "passed": passed,
                "checks": checks,
                **response,
            })

            print(
                f"{name}: {'PASS' if passed else 'FAIL'}"
                f" | {response['response_mode']}"
                f" | {response['latency_seconds']:.4f}s"
            )

    client.flush()

    passed_count = sum(item["passed"] for item in results)

    report = {
        "total_cases": len(results),
        "passed_cases": passed_count,
        "pass_rate": passed_count / len(results),
        "total_tokens": sum(
            item["usage_metadata"].get("total_tokens", 0)
            for item in results
        ),
        "results": results,
    }

    output = Path("presentation_eval_results.json")
    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nPassed: {passed_count}/{len(results)}")
    print(f"Pass rate: {report['pass_rate']:.0%}")
    print(f"Total tokens: {report['total_tokens']}")
    print(f"Report saved: {output}")

    if passed_count != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
