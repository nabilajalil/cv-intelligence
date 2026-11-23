
"""Lesson 6B.9 — Grounded multimodal answer safety tests.

Tests:
- Claim validation.
- Source and candidate ownership.
- Visual verification status.
- Required skill coverage.
- Evidence-grounded answer rendering.
- Retrieval behavior.

No OpenAI API calls.

Run:
    python -m unittest \
        tests.test_multimodal_grounded_answers -v
"""

import copy
import unittest

from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
)

from scripts.multimodal.demo_multimodal_evidence import (
    search_with_evidence,
)

from scripts.multimodal.demo_grounded_multimodal_answers import (
    build_source_context,
    evaluate_candidate,
    render_answer,
)

from scripts.multimodal.multimodal_claim_safety_gate import (
    make_claim,
    validate_claim,
)


class TestMultimodalGroundedAnswers(
    unittest.TestCase
):
    """Regression tests for multimodal claim safety."""

    @classmethod
    def setUpClass(cls):
        cls.records = load_profiles()

        cls.context = build_source_context()

        cls.source_profile = cls.context[
            "source_profile"
        ]

        cls.lookup = cls.context[
            "verification_lookup"
        ]

        cls.candidate_id = cls.source_profile[
            "profile"
        ]["candidate_id"]

        cls.source_file = cls.source_profile[
            "source_file"
        ]

    def python_claim(self):
        """Return a known visually supported claim."""

        return make_claim(
            candidate_id=self.candidate_id,
            field="skills",
            text="Python",
            source_file=self.source_file,
            page_number=1,
        )

    def validate(self, claim, lookup=None):
        """Validate a claim using the test context."""

        return validate_claim(
            claim,
            self.source_profile,
            self.lookup if lookup is None else lookup,
        )

    def search(self, query):
        """Run the real multimodal evidence retriever."""

        return search_with_evidence(
            query,
            self.records,
        )

    def evaluate(
        self,
        query,
        required_skills=None,
        context=None,
    ):
        """Evaluate all candidates for a query."""

        if required_skills is None:
            required_skills = []

        if context is None:
            context = self.context

        candidates = self.search(query)

        return [
            evaluate_candidate(
                candidate,
                context,
                required_skills,
            )
            for candidate in candidates
        ]

    # -----------------------------------------
    # CLAIM VALIDATION
    # -----------------------------------------

    def test_01_valid_python_claim_accepted(self):
        result = self.validate(
            self.python_claim()
        )

        self.assertEqual(
            result["decision"],
            "accept",
        )

    def test_02_wrong_candidate_rejected(self):
        claim = self.python_claim()

        claim["candidate_id"] = "M999"

        result = self.validate(claim)

        self.assertEqual(
            result["decision"],
            "reject",
        )

    def test_03_wrong_source_rejected(self):
        claim = self.python_claim()

        claim["source_file"] = (
            "data/multimodal/input/"
            "synthetic_text_cv.pdf"
        )

        result = self.validate(claim)

        self.assertEqual(
            result["decision"],
            "reject",
        )

    def test_04_wrong_field_rejected(self):
        claim = self.python_claim()

        claim["field"] = "education"

        result = self.validate(claim)

        self.assertEqual(
            result["decision"],
            "reject",
        )

    def test_05_fabricated_skill_rejected(self):
        claim = make_claim(
            candidate_id=self.candidate_id,
            field="skills",
            text="Kubernetes",
            source_file=self.source_file,
        )

        result = self.validate(claim)

        self.assertEqual(
            result["decision"],
            "reject",
        )

    def test_06_unverified_sql_abstains(self):
        claim = make_claim(
            candidate_id=self.candidate_id,
            field="skills",
            text="SQL",
            source_file=self.source_file,
        )

        result = self.validate(claim)

        self.assertEqual(
            result["decision"],
            "abstain",
        )

    def test_07_uncertain_evidence_abstains(self):
        claim = self.python_claim()

        modified_lookup = copy.deepcopy(
            self.lookup
        )

        key = (
            "skills",
            "python",
            self.source_file,
            1,
        )

        self.assertIn(
            key,
            modified_lookup,
        )

        modified_lookup[key] = "uncertain"

        result = self.validate(
            claim,
            lookup=modified_lookup,
        )

        self.assertEqual(
            result["decision"],
            "abstain",
        )

    def test_08_expertise_claim_rejected(self):
        claim = self.python_claim()

        claim["claim_type"] = "expertise"

        result = self.validate(claim)

        self.assertEqual(
            result["decision"],
            "reject",
        )

    # -----------------------------------------
    # REQUIRED SKILL COVERAGE
    # -----------------------------------------

    def test_09_python_docker_both_verified(self):
        evaluations = self.evaluate(
            "Python Docker",
            ["Python", "Docker"],
        )

        eligible = [
            item
            for item in evaluations
            if item["eligible"]
        ]

        self.assertEqual(
            len(eligible),
            1,
        )

        self.assertEqual(
            eligible[0]["candidate_id"],
            self.candidate_id,
        )

        self.assertEqual(
            eligible[0]["missing_verified_skills"],
            [],
        )

    def test_10_unverified_skill_blocks_eligibility(self):
        evaluations = self.evaluate(
            "Python Docker",
            ["Python", "Docker", "SQL"],
        )

        self.assertTrue(evaluations)

        self.assertFalse(
            evaluations[0]["eligible"]
        )

        self.assertIn(
            "SQL",
            evaluations[0][
                "missing_verified_skills"
            ],
        )

    def test_11_no_candidates_for_missing_skills(self):
        evaluations = self.evaluate(
            "Kubernetes Terraform",
            ["Kubernetes", "Terraform"],
        )

        self.assertEqual(
            evaluations,
            [],
        )

    # -----------------------------------------
    # ANSWER RENDERING
    # -----------------------------------------

    def test_12_answer_contains_pdf_provenance(self):
        evaluations = self.evaluate(
            "Python Docker",
            ["Python", "Docker"],
        )

        answer = render_answer(
            "Python Docker",
            evaluations,
        )

        self.assertIn(
            self.source_file,
            answer,
        )

        self.assertIn(
            "page 1",
            answer,
        )

        self.assertIn(
            "CV states (skills): Python",
            answer,
        )

    def test_13_withheld_evidence_not_rendered(self):
        evaluations = self.evaluate(
            "Python Docker",
            ["Python", "Docker"],
        )

        self.assertTrue(evaluations)

        modified = copy.deepcopy(
            evaluations
        )

        modified[0]["withheld_evidence"].append({
            "field": "experience",
            "text": (
                "Worked at Google for five years."
            ),
            "source_file": self.source_file,
            "page_number": 1,
            "decision": "reject",
            "reason": "Fabricated test evidence.",
        })

        answer = render_answer(
            "Python Docker",
            modified,
        )

        self.assertNotIn(
            "Worked at Google",
            answer,
        )

        self.assertIn(
            "Withheld evidence items: 1",
            answer,
        )

    def test_14_ineligible_candidate_not_rendered(self):
        evaluations = self.evaluate(
            "Python Docker",
            ["Python", "Docker", "SQL"],
        )

        answer = render_answer(
            "Python Docker SQL",
            evaluations,
        )

        self.assertIn(
            "No candidates have sufficient "
            "verified evidence",
            answer,
        )

        self.assertNotIn(
            "Candidates with verified CV mentions:",
            answer,
        )

    def test_15_project_query_is_grounded(self):
        evaluations = self.evaluate(
            "Recruitment Analytics Dashboard",
            [],
        )

        eligible = [
            item
            for item in evaluations
            if item["eligible"]
        ]

        self.assertEqual(
            len(eligible),
            1,
        )

        answer = render_answer(
            "Recruitment Analytics Dashboard",
            evaluations,
        )

        self.assertIn(
            "CV states (projects): "
            "Recruitment Analytics Dashboard",
            answer,
        )

        self.assertIn(
            self.source_file,
            answer,
        )


if __name__ == "__main__":
    unittest.main()
