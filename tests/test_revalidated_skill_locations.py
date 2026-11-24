
"""Security tests for revalidated CV skill locations."""

import tempfile
import unittest
from pathlib import Path

from cv_intelligence.prompts.evidence_mapping import (
    build_evidence_mapping,
)
from cv_intelligence.prompts.skill_locations import (
    find_revalidated_skill_mentions,
)
from cv_intelligence.prompts.response_parser import (
    PromptResponseError,
)

from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
)
from scripts.multimodal.demo_multimodal_evidence import (
    search_with_evidence,
)
from scripts.multimodal.demo_protected_grounded_answers import (
    build_protected_context,
    evaluate_protected_candidate,
)


class TestRevalidatedSkillLocations(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.context = build_protected_context()

        retrieved = search_with_evidence(
            "Python Docker",
            load_profiles(),
        )

        evaluations = [
            evaluate_protected_candidate(
                candidate,
                cls.context,
                ["Python", "Docker"],
            )
            for candidate in retrieved
        ]

        cls.mapping = build_evidence_mapping(
            candidates=evaluations,
        )

    def find_python(self, pdf_path=None):
        return find_revalidated_skill_mentions(
            candidate_id="M001",
            skill="Python",
            evidence_mapping=self.mapping,
            source_profile=self.context["source_profile"],
            bound_report=self.context["protected_report"],
            pdf_path=pdf_path,
        )

    def test_valid_source_accepted(self):
        locations = self.find_python()

        self.assertEqual(len(locations), 2)

        sections = {item["section"] for item in locations}

        self.assertEqual(
            sections,
            {"skills", "experience"},
        )

    def test_modified_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            modified_pdf = Path(directory) / "modified.pdf"
            modified_pdf.write_bytes(b"Modified PDF contents")

            with self.assertRaises(PromptResponseError):
                self.find_python(pdf_path=modified_pdf)

    def test_missing_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_pdf = Path(directory) / "missing.pdf"

            with self.assertRaises(PromptResponseError):
                self.find_python(pdf_path=missing_pdf)


if __name__ == "__main__":
    unittest.main()
