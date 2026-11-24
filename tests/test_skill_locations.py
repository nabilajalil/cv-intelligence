
"""Tests for locating skill mentions across CV sections."""

import unittest

from cv_intelligence.prompts.skill_locations import (
    find_all_skill_mentions,
    build_skill_locations,
)


class TestSkillLocations(unittest.TestCase):

    def setUp(self):
        # Synthetic evidence for testing, not actual CV data.
        self.mapping = {
            "verified-1": {
                "candidate_id": "M001",
                "field": "skills",
                "text": "Python, Docker",
                "source_file": "test_cv.pdf",
                "page_number": 1,
            },
            "verified-2": {
                "candidate_id": "M001",
                "field": "experience",
                "text": "Developed REST APIs using Python.",
                "source_file": "test_cv.pdf",
                "page_number": 2,
            },
            "verified-3": {
                "candidate_id": "M001",
                "field": "projects",
                "text": "Built a machine learning project with Python.",
                "source_file": "test_cv.pdf",
                "page_number": 3,
            },
            "verified-4": {
                "candidate_id": "M002",
                "field": "skills",
                "text": "Python",
                "source_file": "other_cv.pdf",
                "page_number": 1,
            },
        }

    def test_python_found_in_three_sections(self):
        locations = find_all_skill_mentions(
            candidate_id="M001",
            skill="Python",
            evidence_mapping=self.mapping,
        )

        sections = {item["section"] for item in locations}

        self.assertEqual(
            sections,
            {"skills", "experience", "projects"},
        )
        self.assertEqual(len(locations), 3)

    def test_docker_found_only_in_skills(self):
        locations = find_all_skill_mentions(
            candidate_id="M001",
            skill="Docker",
            evidence_mapping=self.mapping,
        )

        self.assertEqual(len(locations), 1)
        self.assertEqual(locations[0]["section"], "skills")

    def test_other_candidate_evidence_excluded(self):
        locations = find_all_skill_mentions(
            candidate_id="M001",
            skill="Python",
            evidence_mapping=self.mapping,
        )

        evidence_ids = {
            item["evidence_id"] for item in locations
        }

        self.assertNotIn("verified-4", evidence_ids)

    def test_missing_skill_returns_empty(self):
        locations = find_all_skill_mentions(
            candidate_id="M001",
            skill="Kubernetes",
            evidence_mapping=self.mapping,
        )

        self.assertEqual(locations, [])

    def test_validated_claim_locations(self):
        claims = [
            {
                "candidate_id": "M001",
                "skill": "Python",
                "evidence_ids": [
                    "verified-1",
                    "verified-2",
                ],
            }
        ]

        results = build_skill_locations(
            validated_claims=claims,
            evidence_mapping=self.mapping,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            [item["section"] for item in results[0]["locations"]],
            ["skills", "experience"],
        )


if __name__ == "__main__":
    unittest.main()
