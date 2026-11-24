
"""Security tests for original CV evidence revalidation."""

import tempfile
import unittest
from pathlib import Path

from cv_intelligence.prompts.evidence_mapping import (
    build_evidence_mapping,
)
from cv_intelligence.prompts.response_parser import (
    PromptResponseError,
)
from cv_intelligence.prompts.source_revalidation import (
    revalidate_mapped_evidence,
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


class TestSourceRevalidation(unittest.TestCase):

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

        cls.evidence_id = next(iter(cls.mapping))
        cls.candidate_id = cls.mapping[
            cls.evidence_id
        ]["candidate_id"]

    def revalidate(self, **overrides):
        arguments = {
            "evidence_id": self.evidence_id,
            "candidate_id": self.candidate_id,
            "evidence_mapping": self.mapping,
            "source_profile": self.context[
                "source_profile"
            ],
            "bound_report": self.context[
                "protected_report"
            ],
        }

        arguments.update(overrides)

        return revalidate_mapped_evidence(**arguments)

    def test_valid_evidence_accepted(self):
        result = self.revalidate()

        self.assertEqual(
            result["candidate_id"],
            self.candidate_id,
        )

    def test_wrong_candidate_rejected(self):
        with self.assertRaises(PromptResponseError):
            self.revalidate(
                candidate_id="WRONG-CANDIDATE"
            )

    def test_unknown_evidence_rejected(self):
        with self.assertRaises(PromptResponseError):
            self.revalidate(
                evidence_id="does-not-exist"
            )

    def test_modified_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            modified_pdf = Path(directory) / "modified.pdf"
            modified_pdf.write_bytes(
                b"This is not the original CV PDF."
            )

            with self.assertRaises(PromptResponseError):
                self.revalidate(
                    pdf_path=modified_pdf
                )

    def test_missing_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_pdf = Path(directory) / "missing.pdf"

            with self.assertRaises(PromptResponseError):
                self.revalidate(
                    pdf_path=missing_pdf
                )


if __name__ == "__main__":
    unittest.main()
