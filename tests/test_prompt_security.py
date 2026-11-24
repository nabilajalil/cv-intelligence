
"""Security regression tests for LLM claim proposals."""

import json
import unittest

from cv_intelligence.prompts.candidate_analysis import (
    build_candidate_messages,
)
from cv_intelligence.prompts.response_parser import (
    PromptResponseError,
)
from cv_intelligence.prompts.validation_pipeline import (
    validate_llm_proposals,
)
from cv_intelligence.prompts.skill_support import (
    validate_skill_support,
)


class TestPromptSecurity(unittest.TestCase):

    def setUp(self):
        self.evidence = [
            {
                "evidence_id": "verified-1",
                "candidate_id": "M001",
                "text": "Developed APIs using Python and Docker.",
            },
            {
                "evidence_id": "verified-2",
                "candidate_id": "M002",
                "text": "Built dashboards using SQL.",
            },
        ]

        self.messages = build_candidate_messages(
            query="Find Python developers",
            evidence=self.evidence,
        )

    def make_response(
        self,
        candidate_id,
        skill,
        evidence_ids,
    ):
        return json.dumps({
            "claims": [{
                "candidate_id": candidate_id,
                "skill": skill,
                "evidence_ids": evidence_ids,
            }]
        })

    def test_valid_python_claim(self):
        raw = self.make_response(
            "M001", "Python", ["verified-1"]
        )

        claims = validate_llm_proposals(
            raw_response=raw,
            messages=self.messages,
        )

        validated = validate_skill_support(
            claims=claims,
            evidence=self.evidence,
        )

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["skill"], "Python")

    def test_cross_candidate_evidence_rejected(self):
        raw = self.make_response(
            "M001", "SQL", ["verified-2"]
        )

        with self.assertRaises(PromptResponseError):
            validate_llm_proposals(
                raw_response=raw,
                messages=self.messages,
            )

    def test_unknown_evidence_rejected(self):
        raw = self.make_response(
            "M001", "Python", ["unknown-id"]
        )

        with self.assertRaises(PromptResponseError):
            validate_llm_proposals(
                raw_response=raw,
                messages=self.messages,
            )

    def test_fabricated_skill_rejected(self):
        raw = self.make_response(
            "M001", "Kubernetes", ["verified-1"]
        )

        claims = validate_llm_proposals(
            raw_response=raw,
            messages=self.messages,
        )

        with self.assertRaises(PromptResponseError):
            validate_skill_support(
                claims=claims,
                evidence=self.evidence,
            )

    def test_duplicate_evidence_ids_rejected(self):
        duplicate_evidence = [
            self.evidence[0],
            self.evidence[0],
        ]

        claims = [{
            "candidate_id": "M001",
            "skill": "Python",
            "evidence_ids": ["verified-1"],
        }]

        with self.assertRaises(PromptResponseError):
            validate_skill_support(
                claims=claims,
                evidence=duplicate_evidence,
            )

    def test_invalid_json_rejected(self):
        with self.assertRaises(PromptResponseError):
            validate_llm_proposals(
                raw_response="not valid JSON",
                messages=self.messages,
            )

    def test_empty_claims_allowed(self):
        claims = validate_llm_proposals(
            raw_response='{"claims": []}',
            messages=self.messages,
        )

        self.assertEqual(claims, [])


if __name__ == "__main__":
    unittest.main()
