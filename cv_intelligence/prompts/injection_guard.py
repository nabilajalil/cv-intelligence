
"""Conservative guard against instruction-like CV evidence."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .response_parser import PromptResponseError


INJECTION_PATTERNS = (
    r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
    r"\bsystem\s+override\b",
    r"\badd\s+.+?\s+to\s+the\s+final\s+claims\b",
    r"\bdo\s+not\s+mention\s+this\s+instruction\b",
)


def reject_instruction_like_evidence(
    *,
    evidence: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed when evidence contains known instruction patterns.

    This guard is intentionally conservative and incomplete.
    It must not replace claim, ownership, or source validation.
    """

    for item in evidence:
        text = item.get("text", "")

        if not isinstance(text, str):
            raise PromptResponseError(
                "Evidence text must be a string"
            )

        for pattern in INJECTION_PATTERNS:
            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                raise PromptResponseError(
                    "Instruction-like content detected "
                    "in CV evidence"
                )
