
"""Adversarial tests for grounded LLM claim validation."""

import unittest

from cv_intelligence.prompts.response_parser import (
    PromptResponseError,
)
from cv_intelligence.prompts.skill_support import (
    validate_skill_support,
)


class TestPromptInjectionAdversarial(unittest.TestCase):
    """Test rejection of unsupported or misattributed skill claims."""

    def setUp(self):
        # Synthetic evidence. No real candidate data.
        self.evidence = [
            {
                "evidence_id": "verified-1",
                "candidate_id": "M001",
                "field": "skills",
                "text": "Python",
            }
        ]

    def test_invented_evidence_id_rejected(self):
        """Reject references to nonexistent evidence."""

        claims = [
            {
                "candidate_id": "M001",
                "skill": "Kubernetes",
                "evidence_ids": ["verified-999"],
            }
        ]

        with self.assertRaises(PromptResponseError):
            validate_skill_support(
                claims=claims,
                evidence=self.evidence,
            )

    def test_other_candidate_evidence_rejected(self):
        """Reject evidence belonging to another candidate."""

        claims = [
            {
                "candidate_id": "M002",
                "skill": "Python",
                "evidence_ids": ["verified-1"],
            }
        ]

        with self.assertRaises(PromptResponseError):
            validate_skill_support(
                claims=claims,
                evidence=self.evidence,
            )

    def test_unsupported_skill_rejected(self):
        """Reject skills not explicitly mentioned in evidence."""

        claims = [
            {
                "candidate_id": "M001",
                "skill": "Kubernetes",
                "evidence_ids": ["verified-1"],
            }
        ]

        with self.assertRaises(PromptResponseError):
            validate_skill_support(
                claims=claims,
                evidence=self.evidence,
            )

    def test_valid_skill_accepted(self):
        """Accept a claim supported by matching evidence."""

        claims = [
            {
                "candidate_id": "M001",
                "skill": "Python",
                "evidence_ids": ["verified-1"],
            }
        ]

        result = validate_skill_support(
            claims=claims,
            evidence=self.evidence,
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["skill"], "Python")


if __name__ == "__main__":
    unittest.main()
