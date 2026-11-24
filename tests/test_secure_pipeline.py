
"""Security regression tests for the combined LLM pipeline."""

import json
import tempfile
import unittest
from pathlib import Path

from cv_intelligence.prompts.evidence_mapping import (
    build_evidence_mapping,
)
from cv_intelligence.prompts.protected_integration import (
    build_verified_candidate_messages,
)
from cv_intelligence.prompts.response_parser import (
    PromptResponseError,
)
from cv_intelligence.prompts.secure_pipeline import (
    validate_secure_llm_response,
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


class TestSecurePipeline(unittest.TestCase):

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

        cls.messages = build_verified_candidate_messages(
            query="Python Docker",
            candidates=evaluations,
        )

        cls.mapping = build_evidence_mapping(
            candidates=evaluations,
        )

        cls.python_id = next(
            evidence_id
            for evidence_id, item in cls.mapping.items()
            if item["field"] == "skills"
            and item["text"].strip().casefold() == "python"
        )

        cls.candidate_id = cls.mapping[
            cls.python_id
        ]["candidate_id"]

    def validate(self, skill="Python", **overrides):
        response = json.dumps({
            "claims": [{
                "candidate_id": self.candidate_id,
                "skill": skill,
                "evidence_ids": [self.python_id],
            }]
        })

        arguments = {
            "raw_response": response,
            "messages": self.messages,
            "evidence_mapping": self.mapping,
            "source_profile": self.context[
                "source_profile"
            ],
            "bound_report": self.context[
                "protected_report"
            ],
        }

        arguments.update(overrides)

        return validate_secure_llm_response(**arguments)

    def test_valid_python_claim(self):
        claims = self.validate()
        self.assertEqual(claims[0]["skill"], "Python")

    def test_fabricated_skill_rejected(self):
        with self.assertRaises(PromptResponseError):
            self.validate(skill="Kubernetes")

    def test_wrong_candidate_rejected(self):
        response = json.dumps({
            "claims": [{
                "candidate_id": "WRONG-CANDIDATE",
                "skill": "Python",
                "evidence_ids": [self.python_id],
            }]
        })

        with self.assertRaises(PromptResponseError):
            self.validate(raw_response=response)

    def test_unknown_evidence_rejected(self):
        response = json.dumps({
            "claims": [{
                "candidate_id": self.candidate_id,
                "skill": "Python",
                "evidence_ids": ["unknown-id"],
            }]
        })

        with self.assertRaises(PromptResponseError):
            self.validate(raw_response=response)

    def test_modified_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            modified_pdf = Path(directory) / "modified.pdf"
            modified_pdf.write_bytes(b"Modified PDF contents")

            with self.assertRaises(PromptResponseError):
                self.validate(pdf_path=modified_pdf)

    def test_missing_pdf_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_pdf = Path(directory) / "missing.pdf"

            with self.assertRaises(PromptResponseError):
                self.validate(pdf_path=missing_pdf)


    def test_modified_prompt_evidence_rejected(self):
        import copy

        modified_messages = copy.deepcopy(self.messages)
        payload = json.loads(modified_messages[1]["content"])

        for item in payload["evidence"]:
            if item["evidence_id"] == self.python_id:
                item["text"] = "Kubernetes"

        modified_messages[1]["content"] = json.dumps(payload)

        with self.assertRaises(PromptResponseError):
            self.validate(messages=modified_messages)

    def test_missing_prompt_evidence_rejected(self):
        import copy

        modified_messages = copy.deepcopy(self.messages)
        payload = json.loads(modified_messages[1]["content"])

        payload["evidence"] = [
            item
            for item in payload["evidence"]
            if item["evidence_id"] != self.python_id
        ]

        modified_messages[1]["content"] = json.dumps(payload)

        with self.assertRaises(PromptResponseError):
            self.validate(messages=modified_messages)



if __name__ == "__main__":
    unittest.main()
