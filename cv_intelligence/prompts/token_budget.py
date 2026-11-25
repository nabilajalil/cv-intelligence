"""Token budgeting for LLM response polishing."""

import json
import tiktoken


MAX_CONTEXT_TOKENS = 2048
RESERVED_PROMPT_TOKENS = 512
MAX_OUTPUT_TOKENS = 256

MAX_INPUT_TOKENS = (
    MAX_CONTEXT_TOKENS
    - RESERVED_PROMPT_TOKENS
    - MAX_OUTPUT_TOKENS
)

ENCODING = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """Estimate token consumption before model inference."""
    return len(
        ENCODING.encode(text, disallowed_special=())
    )


def check_context_budget(context: dict) -> dict:
    """Check whether candidate context fits the token budget."""

    serialized = json.dumps(
        context,
        ensure_ascii=False,
    )

    estimated = estimate_tokens(serialized)

    return {
        "estimated_input_tokens": estimated,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "within_budget": estimated <= MAX_INPUT_TOKENS,
    }
