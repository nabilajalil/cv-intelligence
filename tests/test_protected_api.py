
"""Lesson 7.2 — HTTP tests for protected multimodal search."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from cv_intelligence.api import app
from scripts.multimodal.demo_protected_grounded_answers import (
    PROTECTED_REPORT_PATH,
)


client = TestClient(app)


class TestProtectedAPI(unittest.TestCase):

    def post_search(self, query, required_skills=None):
        return client.post(
            "/multimodal/protected-search",
            json={
                "query": query,
                "required_skills": required_skills or [],
            },
        )

    def test_01_health_endpoint(self):
        response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["status"],
            "ok",
        )

    def test_02_protected_route_registered(self):
        schema = client.get("/openapi.json").json()

        self.assertIn(
            "/multimodal/protected-search",
            schema["paths"],
        )
        self.assertIn(
            "post",
            schema["paths"][
                "/multimodal/protected-search"
            ],
        )

    def test_03_python_docker_search(self):
        response = self.post_search(
            "Python Docker",
            ["Python", "Docker"],
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(body["candidate_count"], 1)

        candidate = body["candidates"][0]

        self.assertEqual(candidate["candidate_id"], "M001")
        self.assertTrue(candidate["eligible"])
        self.assertEqual(
            len(candidate["accepted_evidence"]),
            4,
        )

        for item in candidate["accepted_evidence"]:
            self.assertEqual(item["decision"], "accept")
            self.assertEqual(
                item["integrity_status"],
                "match",
            )
            self.assertEqual(item["page_number"], 1)
            self.assertTrue(item["source_file"])

    def test_04_dashboard_search(self):
        response = self.post_search(
            "Recruitment Analytics Dashboard"
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(body["candidate_count"], 1)
        self.assertEqual(
            len(
                body["candidates"][0][
                    "accepted_evidence"
                ]
            ),
            2,
        )

    def test_05_unsupported_skills(self):
        response = self.post_search(
            "Kubernetes Terraform",
            ["Kubernetes", "Terraform"],
        )

        self.assertEqual(response.status_code, 200)

        body = response.json()

        self.assertEqual(body["candidate_count"], 0)
        self.assertEqual(body["candidates"], [])

    def test_06_blank_query_rejected(self):
        response = self.post_search("")

        self.assertEqual(response.status_code, 422)

    def test_07_missing_query_rejected(self):
        response = client.post(
            "/multimodal/protected-search",
            json={},
        )

        self.assertEqual(response.status_code, 422)

    def test_08_long_query_rejected(self):
        response = self.post_search("x" * 501)

        self.assertEqual(response.status_code, 422)

    def test_09_wrong_method_rejected(self):
        response = client.get(
            "/multimodal/protected-search"
        )

        self.assertEqual(response.status_code, 405)

    def test_10_modified_pdf_withheld(self):
        with tempfile.TemporaryDirectory() as tmp:
            altered_pdf = Path(tmp) / "modified.pdf"

            source_pdf = Path(
                "data/multimodal/input/"
                "synthetic_scanned_cv.pdf"
            )

            altered_pdf.write_bytes(
                source_pdf.read_bytes() + b"tampered"
            )

            # Replace the trusted PDF path inside the
            # hash-enforcement module, not the request.
            with patch(
                "scripts.multimodal.demo_protected_grounded_answers."
                "validate_hash_enforced_claim",
                wraps=self._modified_pdf_validator(
                    altered_pdf
                ),
            ):
                response = self.post_search(
                    "Python Docker",
                    ["Python", "Docker"],
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["candidate_count"],
            0,
        )

    def test_11_missing_pdf_withheld(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_pdf = Path(tmp) / "missing.pdf"

            with patch(
                "scripts.multimodal.demo_protected_grounded_answers."
                "validate_hash_enforced_claim",
                wraps=self._modified_pdf_validator(
                    missing_pdf
                ),
            ):
                response = self.post_search(
                    "Python Docker",
                    ["Python", "Docker"],
                )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["candidate_count"],
            0,
        )

    def test_12_existing_search_registered(self):
        schema = client.get("/openapi.json").json()

        self.assertIn("/search", schema["paths"])
        self.assertIn(
            "post",
            schema["paths"]["/search"],
        )

    @staticmethod
    def _modified_pdf_validator(pdf_path):
        from scripts.multimodal.hash_enforced_evidence import (
            validate_hash_enforced_claim,
        )

        def validate(
            claim,
            source_profile,
            bound_report,
            pdf_path=None,
        ):
            return validate_hash_enforced_claim(
                claim,
                source_profile,
                bound_report,
                pdf_path=pdf_path_override,
            )

        pdf_path_override = pdf_path
        return validate


if __name__ == "__main__":
    unittest.main(verbosity=2)
