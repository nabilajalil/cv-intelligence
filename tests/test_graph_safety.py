
"""Lesson 5G.5: Automated GraphRAG safety regression tests.

Run:
    python -m unittest discover -s tests -p "test_graph_safety.py" -v

These are benchmark-level tests, not a certification
of production-grade privacy or semantic correctness.
"""

import copy
import unittest

from scripts.validate_semantic_claims import (
    validate_claim,
)

from scripts.graphrag.demo_claim_validated_graph_answers import (
    build_claims,
    validate_all_claims,
    render_answer,
)

from scripts.graphrag.demo_privacy_safe_graph_answers import (
    make_safe_registry,
)

from scripts.privacy_safe_evidence import (
    make_display_excerpt,
    EMAIL_PATTERN,
    PHONE_PATTERN,
)


class TestGraphSafety(unittest.TestCase):

    def setUp(self):
        """Create a small synthetic evidence fixture."""

        self.selected = ["r_test01"]
        self.required_skills = ["Python", "SQL"]

        self.chunk_lookup = {
            "r_test01_chunk_000": {
                "chunk_id": "r_test01_chunk_000",
                "candidate_id": "r_test01",
                "text": (
                    "Skills: Python SQL Docker\n"
                    "Email: sample@example.com\n"
                    "Phone: +1 415 555 2671\n"
                    "Experience: Software Engineer"
                ),
            }
        }

        self.original_registry = {
            "ev_python": {
                "evidence_id": "ev_python",
                "candidate_id": "r_test01",
                "type": "skill",
                "subject": "Python",
                "chunk_id": "r_test01_chunk_000",
                "quote": "Skills: Python SQL Docker",
            },
            "ev_sql": {
                "evidence_id": "ev_sql",
                "candidate_id": "r_test01",
                "type": "skill",
                "subject": "SQL",
                "chunk_id": "r_test01_chunk_000",
                "quote": "Skills: Python SQL Docker",
            },
        }

        self.candidate_map = {
            "r_test01": ["ev_python", "ev_sql"]
        }

        self.safe_registry = make_safe_registry(
            self.original_registry,
            self.chunk_lookup,
        )

        self.python_claim = {
            "candidate_id": "r_test01",
            "claim_type": "skill_mentioned",
            "skill": "Python",
            "evidence_id": "ev_python",
        }

    def test_supported_skill_claim(self):
        validated, reason = validate_claim(
            self.python_claim,
            self.safe_registry,
            self.selected,
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(validated)

        self.assertEqual(
            validated["statement"],
            (
                "The CV for candidate r_test01 "
                "explicitly mentions Python."
            ),
        )

    def test_unsupported_professional_experience(self):
        claim = {
            **self.python_claim,
            "claim_type": "professional_experience",
        }

        validated, reason = validate_claim(
            claim,
            self.safe_registry,
            self.selected,
        )

        self.assertIsNone(validated)

        self.assertEqual(
            reason,
            "UNSUPPORTED_CLAIM_TYPE",
        )

    def test_unsupported_years_of_experience(self):
        claim = {
            **self.python_claim,
            "claim_type": "years_experience",
            "years": 5,
        }

        validated, reason = validate_claim(
            claim,
            self.safe_registry,
            self.selected,
        )

        self.assertIsNone(validated)
        self.assertEqual(
            reason,
            "UNSUPPORTED_CLAIM_TYPE",
        )

    def test_missing_evidence_id(self):
        claim = {
            **self.python_claim,
            "evidence_id": "ev_missing",
        }

        validated, reason = validate_claim(
            claim,
            self.safe_registry,
            self.selected,
        )

        self.assertIsNone(validated)
        self.assertEqual(
            reason,
            "EVIDENCE_NOT_FOUND",
        )

    def test_candidate_ownership(self):
        claim = {
            **self.python_claim,
            "candidate_id": "r_test02",
        }

        selected = ["r_test01", "r_test02"]

        validated, reason = validate_claim(
            claim,
            self.safe_registry,
            selected,
        )

        self.assertIsNone(validated)
        self.assertEqual(
            reason,
            "EVIDENCE_CANDIDATE_MISMATCH",
        )

    def test_skill_mismatch(self):
        claim = {
            **self.python_claim,
            "skill": "Kubernetes",
        }

        validated, reason = validate_claim(
            claim,
            self.safe_registry,
            self.selected,
        )

        self.assertIsNone(validated)
        self.assertEqual(
            reason,
            "SKILL_MISMATCH",
        )

    def test_wrong_evidence_type(self):
        registry = copy.deepcopy(
            self.safe_registry
        )

        registry["ev_python"]["type"] = "employment"

        validated, reason = validate_claim(
            self.python_claim,
            registry,
            self.selected,
        )

        self.assertIsNone(validated)
        self.assertEqual(
            reason,
            "WRONG_EVIDENCE_TYPE",
        )

    def test_free_text_overclaim_is_discarded(self):
        claim = {
            **self.python_claim,
            "statement": (
                "Expert Python engineer with "
                "10 years of experience."
            ),
        }

        validated, reason = validate_claim(
            claim,
            self.safe_registry,
            self.selected,
        )

        self.assertIsNone(reason)

        self.assertNotIn(
            "10 years",
            validated["statement"],
        )

        self.assertNotIn(
            "Expert",
            validated["statement"],
        )

    def test_safe_excerpts_hide_contact_details(self):
        for record in self.safe_registry.values():
            excerpt = record["display_excerpt"]

            self.assertIsNone(
                EMAIL_PATTERN.search(excerpt)
            )

            self.assertIsNone(
                PHONE_PATTERN.search(excerpt)
            )

            self.assertNotIn(
                "sample@example.com",
                excerpt,
            )

            self.assertNotIn(
                "415 555 2671",
                excerpt,
            )

    def test_skill_excerpt_retains_skill(self):
        for record in self.safe_registry.values():
            excerpt = record["display_excerpt"]

            self.assertIn(
                record["subject"].lower(),
                excerpt.lower(),
            )

    def test_build_and_validate_all_claims(self):
        proposed = build_claims(
            self.selected,
            self.required_skills,
            self.safe_registry,
            self.candidate_map,
        )

        self.assertEqual(len(proposed), 2)

        accepted, rejected = validate_all_claims(
            proposed,
            self.safe_registry,
            self.selected,
        )

        self.assertEqual(len(accepted), 2)
        self.assertEqual(len(rejected), 0)

    def test_render_canonical_answer(self):
        proposed = build_claims(
            self.selected,
            self.required_skills,
            self.safe_registry,
            self.candidate_map,
        )

        accepted, rejected = validate_all_claims(
            proposed,
            self.safe_registry,
            self.selected,
        )

        self.assertEqual(len(rejected), 0)

        answer = render_answer(
            self.selected,
            self.required_skills,
            accepted,
            self.safe_registry,
        )

        self.assertIn(
            "explicitly mentions Python",
            answer,
        )

        self.assertIn(
            "explicitly mentions SQL",
            answer,
        )

        self.assertNotIn(
            "ev_python",
            answer,
        )

        self.assertNotIn(
            "sample@example.com",
            answer,
        )

    def test_no_match_fallback(self):
        answer = render_answer(
            selected=[],
            required_skills=[
                "Python",
                "SQL",
                "Linux",
                "GitLab",
                "JavaScript",
            ],
            accepted=[],
            safe_registry={},
        )

        self.assertEqual(
            answer,
            (
                "No candidates were verified as mentioning "
                "all required skills."
            ),
        )


if __name__ == "__main__":
    unittest.main()
