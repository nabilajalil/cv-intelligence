
"""Repeatable evaluation of local Qwen CV skill proposals."""

import json

from scripts.llm.prompt_eval_cases import EVAL_CASES
from cv_intelligence.prompts.candidate_analysis import (
    build_candidate_messages,
)
from cv_intelligence.prompts.ollama_client import (
    generate_local_response,
)
from cv_intelligence.prompts.response_parser import (
    PromptResponseError,
    parse_claim_proposals,
)
from cv_intelligence.prompts.validation_pipeline import (
    validate_llm_proposals,
)
from cv_intelligence.prompts.skill_support import (
    validate_skill_support,
)


def main():
    results = []

    for case in EVAL_CASES:
        print(f"\nRUNNING: {case['id']}")

        messages = build_candidate_messages(
            query=case["query"],
            evidence=case["evidence"],
        )

        result = {
            "id": case["id"],
            "category": case["category"],
            "expected_skills": case["expected_skills"],
            "raw_skills": [],
            "accepted_skills": [],
            "raw_unsupported_skills": [],
            "missing_expected_skills": [],
            "validation_status": "not_run",
            "error": None,
        }

        try:
            raw_response = generate_local_response(messages)

            # Parse the LLM proposal separately from security validation.
            proposals = parse_claim_proposals(
                raw_response,
                allowed_evidence_ids={
                    item["evidence_id"]
                    for item in case["evidence"]
                },
                allowed_candidate_ids={
                    item["candidate_id"]
                    for item in case["evidence"]
                },
            )

            result["raw_skills"] = sorted({
                claim["skill"] for claim in proposals
            })

            expected = set(case["expected_skills"])
            proposed = set(result["raw_skills"])

            result["raw_unsupported_skills"] = sorted(
                proposed - expected
            )
            result["missing_expected_skills"] = sorted(
                expected - proposed
            )

            claims = validate_llm_proposals(
                raw_response=raw_response,
                messages=messages,
            )

            claims = validate_skill_support(
                claims=claims,
                evidence=case["evidence"],
            )

            result["accepted_skills"] = sorted({
                claim["skill"] for claim in claims
            })
            result["validation_status"] = "accepted"

        except (PromptResponseError, ValueError, TypeError) as exc:
            result["validation_status"] = "rejected"
            result["error"] = str(exc)

        except Exception as exc:
            result["validation_status"] = "execution_error"
            result["error"] = f"{type(exc).__name__}: {exc}"

        results.append(result)

        print(json.dumps(result, indent=2))

    print("\n" + "=" * 50)
    print("PROMPT EVALUATION SUMMARY")
    print("=" * 50)

    for result in results:
        print(
            f"{result['id']}: "
            f"{result['validation_status']}, "
            f"unsupported={result['raw_unsupported_skills']}, "
            f"missing={result['missing_expected_skills']}"
        )

    with open(
        "prompt_eval_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\nSaved results to prompt_eval_results.json")


if __name__ == "__main__":
    main()
