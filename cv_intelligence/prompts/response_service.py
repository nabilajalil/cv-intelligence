"""Orchestrate constrained response polishing and fallback."""

import json
import time

from .response_fallback import build_fallback_response
from .response_context import build_response_context
from .presentation_budget import check_presentation_budget
from .presentation_chain import build_presentation_chain
from .presentation_prompt import PROMPT_VERSION
from .response_renderer import render_response
from .response_validation import PresentationChoice


def polish_candidate_response(
    candidate: dict,
    *,
    chain=None,
) -> dict:
    """Polish presentation without modifying evaluated facts."""

    start = time.perf_counter()

    fallback = build_fallback_response(candidate)

    result = {
        "summary": fallback,
        "response_mode": "deterministic_fallback",
        "fallback_reason": None,
        "prompt_version": PROMPT_VERSION,
        "selected_style": None,
        "token_budget": {},
        "usage_metadata": {},
    }

    try:
        if candidate["required_skill_count"] == 0:
            result["summary"] = (
                "No explicit technical skills were requested. "
                "Skill matching was not performed."
            )
            result["fallback_reason"] = "no_explicit_skills"
            return result

        context = build_response_context(candidate)

        budget = check_presentation_budget(context)

        result["token_budget"] = budget

        if not budget["within_budget"]:
            result["fallback_reason"] = "token_budget_exceeded"
            return result

        if chain is None:
            chain = build_presentation_chain()

        generated = chain.invoke({
            "context": json.dumps(
                context,
                ensure_ascii=False,
            )
        })

        choice = PresentationChoice.model_validate(
            generated["choice"]
        )

        result["usage_metadata"] = generated.get(
            "usage_metadata", {}
        )

        summary = render_response(context, choice)

        result.update({
            "summary": summary,
            "response_mode": "llm_assisted",
            "selected_style": choice.style,
        })

    except Exception as exc:
        result["fallback_reason"] = type(exc).__name__

    finally:
        result["latency_seconds"] = round(
            time.perf_counter() - start,
            4,
        )

    return result
