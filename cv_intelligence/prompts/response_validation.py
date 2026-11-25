"""Strict validation for LLM presentation choices."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class PresentationChoice(BaseModel):
    """Restrict LLM output to approved presentation styles."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    style: Literal[
        "professional",
        "concise",
        "evidence_first",
    ]
