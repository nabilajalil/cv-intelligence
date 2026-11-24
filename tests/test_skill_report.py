
"""Tests for the recruiter-friendly skill report."""

import unittest

from cv_intelligence.prompts.skill_report import format_skill_report


class TestSkillReport(unittest.TestCase):

    def test_skill_in_multiple_sections(self):
        report = format_skill_report(
            candidate_id="M001",
            skill_locations={
                "Python": [
                    {
                        "section": "skills",
                        "page_number": 1,
                        "text": "Python",
                    },
                    {
                        "section": "experience",
                        "page_number": 2,
                        "text": "Built a Python pipeline.",
                    },
                    {
                        "section": "projects",
                        "page_number": 3,
                        "text": "Created a Python dashboard.",
                    },
                ],
            },
        )

        self.assertIn("Candidate M001", report)
        self.assertIn("Python", report)
        self.assertIn("Skills (Page 1)", report)
        self.assertIn("Experience (Page 2)", report)
        self.assertIn("Projects (Page 3)", report)
        self.assertIn("Built a Python pipeline.", report)

    def test_skill_without_mentions(self):
        report = format_skill_report(
            candidate_id="M001",
            skill_locations={"Kubernetes": []},
        )

        self.assertIn("Kubernetes", report)
        self.assertIn("No verified mentions found.", report)

    def test_multiple_skills(self):
        report = format_skill_report(
            candidate_id="M001",
            skill_locations={
                "Python": [
                    {
                        "section": "skills",
                        "page_number": 1,
                        "text": "Python",
                    },
                ],
                "Docker": [
                    {
                        "section": "experience",
                        "page_number": 2,
                        "text": "Used Docker.",
                    },
                ],
            },
        )

        self.assertIn("Python", report)
        self.assertIn("Docker", report)
        self.assertIn("Skills (Page 1)", report)
        self.assertIn("Experience (Page 2)", report)


if __name__ == "__main__":
    unittest.main()
