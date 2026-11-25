"""Pre-inference budget checks for presentation selection."""

import json

from .presentation_prompt import PRESENTATION_PROMPT
from .presentation_chain import MAX_PRESENTATION_OUTPUT_TOKENS
from .token_budget import (
    MAX_CONTEXT_TOKENS,
    check_context_budget,
    estimate_tokens,
)


CHAT_OVERHEAD_RESERVE = 128


def check_presentation_budget(context: dict) -> dict:
    """Estimate whether the complete prompt fits the budget."""

    serialized = json.dumps(context, ensure_ascii=False)

    context_budget = check_context_budget(context)

    messages = PRESENTATION_PROMPT.format_messages(
        context=serialized
    )

    estimated_prompt_tokens = sum(
        estimate_tokens(message.content)
        for message in messages
    )

    estimated_total_tokens = (
        estimated_prompt_tokens
        + CHAT_OVERHEAD_RESERVE
        + MAX_PRESENTATION_OUTPUT_TOKENS
    )

    return {
        "estimated_context_tokens": (
            context_budget["estimated_input_tokens"]
        ),
        "estimated_prompt_tokens": estimated_prompt_tokens,
        "reserved_chat_overhead": CHAT_OVERHEAD_RESERVE,
        "reserved_output_tokens": MAX_PRESENTATION_OUTPUT_TOKENS,
        "estimated_total_tokens": estimated_total_tokens,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "within_budget": (
            context_budget["within_budget"]
            and estimated_total_tokens <= MAX_CONTEXT_TOKENS
        ),
    }
