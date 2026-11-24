
"""Experimental local Qwen integration with protected CV evidence."""

from cv_intelligence.prompts.evidence_mapping import (
    build_evidence_mapping,
)
from cv_intelligence.prompts.ollama_client import (
    generate_local_response,
)
from cv_intelligence.prompts.protected_integration import (
    build_verified_candidate_messages,
)
from cv_intelligence.prompts.secure_pipeline import (
    validate_secure_llm_response,
)
from cv_intelligence.prompts.response_parser import (
    PromptResponseError,
)
from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
)
from scripts.multimodal.demo_multimodal_evidence import (
    search_with_evidence,
)
from scripts.multimodal.demo_protected_grounded_answers import (
    build_protected_context,
    evaluate_protected_candidate,
)


def main():
    query = "Which candidates have Python and Docker skills?"
    required_skills = ["Python", "Docker"]

    # Step 1: Retrieve CV evidence.
    context = build_protected_context()
    retrieved = search_with_evidence(
        "Python Docker",
        load_profiles(),
    )

    # Step 2: Apply the existing protected candidate evaluation.
    evaluations = [
        evaluate_protected_candidate(
            candidate,
            context,
            required_skills,
        )
        for candidate in retrieved
    ]

    # Step 3: Build the LLM prompt and evidence mapping.
    messages = build_verified_candidate_messages(
        query=query,
        candidates=evaluations,
    )
    evidence_mapping = build_evidence_mapping(
        candidates=evaluations,
    )

    print("Verified evidence records:", len(evidence_mapping))

    # Step 4: Ask the local model for proposed claims.
    raw_response = generate_local_response(messages)

    print("\nUntrusted Qwen response:")
    print(raw_response)

    # Step 5: Validate against the original protected evidence.
    try:
        validated_claims = validate_secure_llm_response(
            raw_response=raw_response,
            messages=messages,
            evidence_mapping=evidence_mapping,
            source_profile=context["source_profile"],
            bound_report=context["protected_report"],
        )
    except PromptResponseError as exc:
        print("\nSECURITY VALIDATION: REJECTED")
        print("Reason:", exc)
        return

    print("\nSECURITY VALIDATION: PASSED")
    print("Validated claims:")
    for claim in validated_claims:
        print(claim)


if __name__ == "__main__":
    main()
