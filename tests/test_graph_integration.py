
"""Lesson 5G.6: Saved GraphRAG artifact integration tests.

Run:
    python -m unittest discover -s tests \
        -p "test_graph_integration.py" -v

Tests the two saved Lesson 5G.4 benchmark outputs.
No OpenAI calls are required.

These tests do not certify production-grade anonymization.
"""

import json
import re
import unittest
from pathlib import Path

from scripts.validate_semantic_claims import (
    validate_claim,
)

from scripts.graphrag.demo_claim_validated_graph_answers import (
    render_answer,
)

from scripts.privacy_safe_evidence import (
    EMAIL_PATTERN,
    PHONE_PATTERN,
    DOB_PATTERN,
)


OUTPUT_DIR = Path(
    "data/public_benchmark/graph/claim_validated_answers"
)

MATCH_FILE = OUTPUT_DIR / "cf54dc156415.json"
NO_MATCH_FILE = OUTPUT_DIR / "0bce9bad1b45.json"

EXPECTED_CANDIDATES = [
    "r00001",
    "r00041",
    "r00038",
    "r00049",
    "r00053",
]

EXPECTED_SKILLS = ["Python", "SQL"]


def load_artifact(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing artifact: {path}\n"
            "Run Lesson 5G.4 for this query first."
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def has_contact_pattern(text):
    """Check known contact/identity patterns.

    This is a limited heuristic, not comprehensive PII
    detection or anonymization.
    """

    return any(
        pattern.search(text)
        for pattern in (
            EMAIL_PATTERN,
            PHONE_PATTERN,
            DOB_PATTERN,
        )
    )


class TestGraphIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.match = load_artifact(MATCH_FILE)
        cls.no_match = load_artifact(NO_MATCH_FILE)

    def test_expected_candidates(self):
        self.assertEqual(
            self.match["selected_candidates"],
            EXPECTED_CANDIDATES,
        )

    def test_expected_skills(self):
        self.assertEqual(
            self.match["required_skills"],
            EXPECTED_SKILLS,
        )

    def test_complete_skill_coverage(self):
        coverage = self.match["skill_coverage"]

        self.assertEqual(
            coverage["covered"],
            10,
        )

        self.assertEqual(
            coverage["expected"],
            10,
        )

    def test_claim_counts(self):
        self.assertEqual(
            len(self.match["proposed_claims"]),
            10,
        )

        self.assertEqual(
            len(self.match["accepted_claims"]),
            10,
        )

        self.assertEqual(
            len(self.match["rejected_claims"]),
            0,
        )

    def test_all_claims_are_revalidated(self):
        registry = self.match[
            "safe_evidence_registry"
        ]

        selected = self.match[
            "selected_candidates"
        ]

        for claim in self.match["accepted_claims"]:
            with self.subTest(
                candidate=claim["candidate_id"],
                skill=claim["skill"],
            ):
                validated, reason = validate_claim(
                    claim,
                    registry,
                    selected,
                )

                self.assertIsNone(reason)
                self.assertIsNotNone(validated)

                self.assertEqual(
                    validated,
                    claim,
                )

    def test_evidence_ids_match_registry_keys(self):
        registry = self.match[
            "safe_evidence_registry"
        ]

        for evidence_id, record in registry.items():
            with self.subTest(evidence_id=evidence_id):
                self.assertEqual(
                    record["evidence_id"],
                    evidence_id,
                )

    def test_claim_evidence_ownership(self):
        registry = self.match[
            "safe_evidence_registry"
        ]

        for claim in self.match["accepted_claims"]:
            evidence = registry[
                claim["evidence_id"]
            ]

            self.assertEqual(
                evidence["candidate_id"],
                claim["candidate_id"],
            )

            self.assertEqual(
                evidence["chunk_id"],
                claim["chunk_id"],
            )

            self.assertEqual(
                evidence["subject"].casefold(),
                claim["skill"].casefold(),
            )

    def test_every_selected_candidate_has_both_skills(self):
        accepted = self.match["accepted_claims"]

        for candidate_id in EXPECTED_CANDIDATES:
            skills = {
                claim["skill"].casefold()
                for claim in accepted
                if claim["candidate_id"] == candidate_id
            }

            self.assertEqual(
                skills,
                {"python", "sql"},
            )

    def test_no_contact_patterns_in_display_excerpts(self):
        registry = self.match[
            "safe_evidence_registry"
        ]

        for evidence_id, record in registry.items():
            excerpt = record.get("display_excerpt")

            if excerpt is None:
                continue

            with self.subTest(evidence_id=evidence_id):
                self.assertFalse(
                    has_contact_pattern(excerpt),
                    f"Possible PII in {evidence_id}",
                )

    def test_no_contact_patterns_in_final_answer(self):
        self.assertFalse(
            has_contact_pattern(
                self.match["answer"]
            )
        )

        self.assertFalse(
            has_contact_pattern(
                self.no_match["answer"]
            )
        )

    def test_no_internal_evidence_ids_in_final_answer(self):
        answer = self.match["answer"]

        self.assertIsNone(
            re.search(
                r"\bev_[a-zA-Z0-9_]+\b",
                answer,
            )
        )

    def test_final_answer_reconstructs_exactly(self):
        reconstructed = render_answer(
            self.match["selected_candidates"],
            self.match["required_skills"],
            self.match["accepted_claims"],
            self.match["safe_evidence_registry"],
        )

        self.assertEqual(
            reconstructed,
            self.match["answer"],
        )

    def test_no_match_has_no_candidates(self):
        self.assertEqual(
            self.no_match["selected_candidates"],
            [],
        )

    def test_no_match_has_no_claims(self):
        for key in (
            "proposed_claims",
            "accepted_claims",
            "rejected_claims",
        ):
            self.assertEqual(
                self.no_match[key],
                [],
            )

    def test_no_match_has_no_evidence(self):
        self.assertEqual(
            self.no_match["safe_evidence_registry"],
            {},
        )

        coverage = self.no_match["skill_coverage"]

        self.assertEqual(
            coverage["covered"],
            0,
        )

        self.assertEqual(
            coverage["expected"],
            0,
        )

    def test_no_match_fallback_is_exact(self):
        self.assertEqual(
            self.no_match["answer"],
            (
                "No candidates were verified as mentioning "
                "all required skills."
            ),
        )

    def test_no_match_answer_reconstructs(self):
        reconstructed = render_answer(
            self.no_match["selected_candidates"],
            self.no_match["required_skills"],
            self.no_match["accepted_claims"],
            self.no_match["safe_evidence_registry"],
        )

        self.assertEqual(
            reconstructed,
            self.no_match["answer"],
        )


if __name__ == "__main__":
    unittest.main()
