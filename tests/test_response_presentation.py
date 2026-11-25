"""Regression tests for constrained response presentation."""

import pytest
from pydantic import ValidationError

from cv_intelligence.prompts.response_service import (
    polish_candidate_response,
)
from cv_intelligence.prompts.response_validation import (
    PresentationChoice,
)


@pytest.fixture
def candidate():
    return {
        "candidate_id": "synthetic-001",
        "document_id": "synthetic-doc",
        "best_rrf_score": 0.032,
        "matched_skills": ["Python", "Docker"],
        "skills_not_found_in_evidence": ["Kubernetes"],
        "matched_skill_count": 2,
        "required_skill_count": 3,
        "skill_evidence": {
            "Python": [{"section": "Skills"}],
            "Docker": [{"section": "Projects"}],
        },
    }


class MockChain:
    def __init__(self, style):
        self.style = style

    def invoke(self, payload):
        return {
            "choice": {"style": self.style},
            "usage_metadata": {
                "input_tokens": 240,
                "output_tokens": 10,
                "total_tokens": 250,
            },
        }


class FailingChain:
    def invoke(self, payload):
        raise ConnectionError("Simulated Ollama failure")


class ForbiddenChain:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        raise AssertionError("Inference should not run")


@pytest.mark.parametrize(
    "style",
    ["professional", "concise", "evidence_first"],
)
def test_valid_presentation(candidate, style):
    result = polish_candidate_response(
        candidate, chain=MockChain(style)
    )

    assert result["response_mode"] == "llm_assisted"
    assert result["selected_style"] == style
    assert result["usage_metadata"]["total_tokens"] == 250

    summary = result["summary"]

    assert "Python" in summary
    assert "Docker" in summary
    assert "2 of 3 requested skills" in summary
    assert "No explicit mention of Kubernetes was found" in summary
    assert "independently verified proficiency" in summary


@pytest.mark.parametrize(
    "payload",
    [
        {"style": "unsupported"},
        {"style": "professional", "summary": "Invented claim"},
        {"style": 123},
        {},
    ],
)
def test_invalid_schema(payload):
    with pytest.raises(ValidationError):
        PresentationChoice.model_validate(payload)


def test_invalid_output_fallback(candidate):
    result = polish_candidate_response(
        candidate, chain=MockChain("unsupported")
    )

    assert result["response_mode"] == "deterministic_fallback"
    assert result["fallback_reason"] == "ValidationError"
    assert "2 of 3 requested skills" in result["summary"]


def test_connection_failure_fallback(candidate):
    result = polish_candidate_response(
        candidate, chain=FailingChain()
    )

    assert result["response_mode"] == "deterministic_fallback"
    assert result["fallback_reason"] == "ConnectionError"
    assert "Python" in result["summary"]
    assert "No explicit mention of Kubernetes" in result["summary"]


def test_oversized_context_skips_inference(candidate):
    oversized = "Projects " * 5000

    candidate["skill_evidence"]["Python"] = [
        {"section": oversized}
    ]

    chain = ForbiddenChain()

    result = polish_candidate_response(
        candidate, chain=chain
    )

    assert result["fallback_reason"] == "token_budget_exceeded"
    assert result["response_mode"] == "deterministic_fallback"
    assert chain.calls == 0
    assert oversized not in result["summary"]
    assert len(result["summary"]) < 1000


def test_no_explicit_skills_skips_inference(candidate):
    candidate.update({
        "matched_skills": [],
        "skills_not_found_in_evidence": [],
        "matched_skill_count": 0,
        "required_skill_count": 0,
        "skill_evidence": {},
    })

    chain = ForbiddenChain()

    result = polish_candidate_response(
        candidate, chain=chain
    )

    assert result["fallback_reason"] == "no_explicit_skills"
    assert chain.calls == 0
    assert "Skill matching was not performed" in result["summary"]


def test_agent_preserves_candidate_facts(candidate, monkeypatch):
    from cv_intelligence.agent import summarize_node
    from cv_intelligence.prompts import response_service

    monkeypatch.setenv("CV_ENABLE_LLM_POLISHING", "true")

    monkeypatch.setattr(
        response_service,
        "build_presentation_chain",
        lambda: MockChain("professional"),
    )

    result = summarize_node({
        "evaluated_candidates": [candidate]
    })["evaluated_candidates"][0]

    for key, value in candidate.items():
        assert result[key] == value

    assert (
        result["response_metadata"]["response_mode"]
        == "llm_assisted"
    )
