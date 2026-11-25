"""Tests for experimental LLM-generated recruiter responses."""

from unittest.mock import Mock

import pytest

from cv_intelligence.prompts import response_chain
from cv_intelligence.prompts.response_chain import PolishedResponse
from cv_intelligence.prompts.response_service import (
    polish_candidate_response,
)


@pytest.fixture
def candidate():
    return {
        "candidate_id": "synthetic-001",
        "matched_skills": ["Python", "Docker"],
        "skills_not_found_in_evidence": ["Kubernetes"],
        "matched_skill_count": 2,
        "required_skill_count": 3,
        "skill_evidence": {
            "Python": [{"section": "Skills"}],
            "Docker": [{"section": "Projects"}],
        },
    }


@pytest.fixture
def generator(monkeypatch):
    monkeypatch.setenv("CV_RESPONSE_MODE", "generative")

    chain = Mock()
    chain.invoke.return_value = PolishedResponse(
        summary="Python and Docker are documented in the CV."
    )

    monkeypatch.setattr(
        response_chain,
        "build_response_chain",
        lambda: chain,
    )

    return chain


def test_successful_generation(candidate, generator):
    result = polish_candidate_response(candidate)

    assert result["response_mode"] == "llm_generated_experimental"
    assert "Python and Docker are documented" in result["summary"]
    generator.invoke.assert_called_once()


def test_candidate_facts_preserved(candidate, generator):
    from copy import deepcopy

    original = deepcopy(candidate)

    polish_candidate_response(candidate)

    assert candidate == original


def test_invalid_output_fallback(candidate, generator):
    from langchain_core.exceptions import OutputParserException

    generator.invoke.side_effect = OutputParserException(
        "Invalid JSON"
    )

    result = polish_candidate_response(candidate)

    assert result["response_mode"] == "deterministic_fallback"
    assert result["fallback_reason"] == "OutputParserException"


def test_inference_failure_fallback(candidate, generator):
    generator.invoke.side_effect = ConnectionError(
        "Ollama unavailable"
    )

    result = polish_candidate_response(candidate)

    assert result["response_mode"] == "deterministic_fallback"
    assert result["fallback_reason"] == "ConnectionError"


def test_oversized_context_skips_inference(candidate, generator):
    candidate["skill_evidence"]["Python"] = [
        {"section": "Large context " * 5000}
    ]

    result = polish_candidate_response(candidate)

    assert result["response_mode"] == "deterministic_fallback"
    assert result["fallback_reason"] == "token_budget_exceeded"
    generator.invoke.assert_not_called()


def test_evidence_disclaimer_preserved(candidate, generator):
    result = polish_candidate_response(candidate)

    assert (
        "Documented skill mentions do not establish "
        "independently verified proficiency or experience."
        in result["summary"]
    )
