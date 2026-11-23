
"""Lesson 6C.6 — Protected grounded-answer regression tests.

Run:
    python -m unittest \
        tests.test_protected_grounded_answers -v

No OpenAI API calls.
"""

import copy
import tempfile
import unittest
from pathlib import Path

from scripts.multimodal.demo_multimodal_retrieval import (
    load_profiles,
)
from scripts.multimodal.demo_multimodal_evidence import (
    search_with_evidence,
)
from scripts.multimodal.demo_grounded_multimodal_answers import (
    make_evidence_claim,
    render_answer,
)
from scripts.demo_protected_grounded_answers import (
    PROTECTED_REPORT_PATH,
    build_protected_context,
    evaluate_protected_candidate,
    prepare_protected_report,
    run_query,
)
from scripts.hash_enforced_evidence import (
    validate_hash_enforced_claim,
)
from scripts.multimodal.multimodal_claim_safety_gate import (
    load_json,
    make_claim,
)


class ProtectedGroundedAnswerTests(
    unittest.TestCase
):
    """Test the protected multimodal answer pipeline."""

    @classmethod
    def setUpClass(cls):
        cls.records = load_profiles()
        cls.context = build_protected_context()

        cls.source_profile = cls.context[
            "source_profile"
        ]

        cls.protected_report = cls.context[
            "protected_report"
        ]

        cls.pdf_path = Path(
            cls.source_profile["source_file"]
        )

    def get_valid_claim(self):
        """Create a known verified Docker claim."""

        return make_claim(
            candidate_id="M001",
            field="skills",
            text="Docker",
            source_file=self.source_profile[
                "source_file"
            ],
            page_number=1,
            claim_type="cv_mention",
        )

    def test_01_protected_report_loads(self):
        """The live protected report is available."""

        report = load_json(
            PROTECTED_REPORT_PATH
        )

        self.assertEqual(
            report["verification_type"],
            "pre_post_fingerprint_protected",
        )

        self.assertEqual(
            report["hash_algorithm"],
            "sha256",
        )

        self.assertEqual(
            len(report["results"]),
            6,
        )

    def test_02_valid_document_accepts_claim(self):
        """Verified evidence passes the hash gate."""

        result = validate_hash_enforced_claim(
            self.get_valid_claim(),
            self.source_profile,
            self.protected_report,
        )

        self.assertEqual(
            result["decision"],
            "accept",
        )

    def test_03_python_docker_query(self):
        """Python Docker returns four accepted items."""

        result = run_query(
            "Python Docker",
            self.records,
            self.context,
        )

        evaluations = result["evaluations"]

        self.assertEqual(
            len(evaluations),
            1,
        )

        self.assertTrue(
            evaluations[0]["eligible"]
        )

        self.assertEqual(
            len(
                evaluations[0][
                    "accepted_evidence"
                ]
            ),
            4,
        )

    def test_04_dashboard_query(self):
        """Dashboard returns two accepted items."""

        result = run_query(
            "Recruitment Analytics Dashboard",
            self.records,
            self.context,
        )

        evaluations = result["evaluations"]

        self.assertEqual(
            len(evaluations),
            1,
        )

        self.assertTrue(
            evaluations[0]["eligible"]
        )

        self.assertEqual(
            len(
                evaluations[0][
                    "accepted_evidence"
                ]
            ),
            2,
        )

    def test_05_no_unsupported_candidate(self):
        """Kubernetes Terraform has no verified match."""

        result = run_query(
            "Kubernetes Terraform",
            self.records,
            self.context,
        )

        self.assertFalse(
            any(
                item["eligible"]
                for item in result["evaluations"]
            )
        )

        self.assertIn(
            "No candidates",
            result["answer"],
        )

    def test_06_modified_pdf_rejects_evidence(self):
        """Modified document bytes fail closed."""

        with tempfile.TemporaryDirectory() as temp:
            modified_path = (
                Path(temp) / "modified.pdf"
            )

            modified_path.write_bytes(
                self.pdf_path.read_bytes()
                + b"\n% test modification\n"
            )

            result = validate_hash_enforced_claim(
                self.get_valid_claim(),
                self.source_profile,
                self.protected_report,
                pdf_path=modified_path,
            )

        self.assertEqual(
            result["decision"],
            "reject",
        )

        self.assertEqual(
            result["reason"],
            "document_fingerprint_mismatch",
        )

    def test_07_missing_pdf_rejects_evidence(self):
        """Missing source documents fail closed."""

        with tempfile.TemporaryDirectory() as temp:
            missing_path = (
                Path(temp) / "missing.pdf"
            )

            result = validate_hash_enforced_claim(
                self.get_valid_claim(),
                self.source_profile,
                self.protected_report,
                pdf_path=missing_path,
            )

        self.assertEqual(
            result["decision"],
            "reject",
        )

        self.assertEqual(
            result["reason"],
            "source_file_missing",
        )

    def test_08_wrong_candidate_rejected(self):
        """Evidence cannot be assigned to another CV."""

        claim = self.get_valid_claim()
        claim["candidate_id"] = "M999"

        result = validate_hash_enforced_claim(
            claim,
            self.source_profile,
            self.protected_report,
        )

        self.assertEqual(
            result["decision"],
            "reject",
        )

    def test_09_fabricated_skill_rejected(self):
        """A fabricated skill cannot be verified."""

        claim = self.get_valid_claim()
        claim["evidence_text"] = "Kubernetes"

        result = validate_hash_enforced_claim(
            claim,
            self.source_profile,
            self.protected_report,
        )

        self.assertEqual(
            result["decision"],
            "reject",
        )

    def test_10_unsupported_evidence_rejected(self):
        """Explicitly unsupported visual evidence fails."""

        report = copy.deepcopy(
            self.protected_report
        )

        for item in report["results"]:
            if (
                item["field"] == "skills"
                and item["text"] == "Docker"
            ):
                item["status"] = "unsupported"

        result = validate_hash_enforced_claim(
            self.get_valid_claim(),
            self.source_profile,
            report,
        )

        self.assertEqual(
            result["decision"],
            "reject",
        )

    def test_11_uncertain_evidence_abstains(self):
        """Uncertain visual evidence must be withheld."""

        report = copy.deepcopy(
            self.protected_report
        )

        for item in report["results"]:
            if (
                item["field"] == "skills"
                and item["text"] == "Docker"
            ):
                item["status"] = "uncertain"

        result = validate_hash_enforced_claim(
            self.get_valid_claim(),
            self.source_profile,
            report,
        )

        self.assertEqual(
            result["decision"],
            "abstain",
        )

    def test_12_mismatched_evidence_hash_blocked(self):
        """Evidence-level fingerprint mismatch is blocked."""

        report = load_json(
            PROTECTED_REPORT_PATH
        )

        report["results"][0][
            "source_sha256"
        ] = "0" * 64

        with self.assertRaises(ValueError):
            prepare_protected_report(
                report
            )


if __name__ == "__main__":
    unittest.main()
