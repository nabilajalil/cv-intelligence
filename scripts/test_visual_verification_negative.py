
"""Lesson 6B.6 — Visual verification negative testing.

Tests whether a vision model:
- Accepts genuine CV evidence.
- Rejects fabricated CV evidence.
- Reports uncertainty when appropriate.

Uses one paid OpenAI API request.

Run:
    python -m scripts.test_visual_verification_negative
"""

import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from scripts.verify_scanned_cv_visual_evidence import (
    MODEL,
    PDF_PATH,
    render_page,
    verify_with_vision,
)


OUTPUT_PATH = Path(
    "data/multimodal/reports/"
    "visual_negative_test_evaluation.json"
)


TEST_CASES = [
    {
        "evidence_id": "P001",
        "field": "skills",
        "text": "Python",
        "expected_status": "supported",
    },
    {
        "evidence_id": "P002",
        "field": "experience",
        "text": (
            "Built a Python data-processing pipeline."
        ),
        "expected_status": "supported",
    },
    {
        "evidence_id": "P003",
        "field": "projects",
        "text": "Recruitment Analytics Dashboard",
        "expected_status": "supported",
    },
    {
        "evidence_id": "N001",
        "field": "experience",
        "text": (
            "Experienced with Kubernetes and Terraform."
        ),
        "expected_status": "unsupported",
    },
    {
        "evidence_id": "N002",
        "field": "experience",
        "text": (
            "Worked at Google for five years."
        ),
        "expected_status": "unsupported",
    },
    {
        "evidence_id": "N003",
        "field": "education",
        "text": (
            "Holds a PhD in Artificial Intelligence."
        ),
        "expected_status": "unsupported",
    },
]


def evaluate_decisions(decisions):
    """Compare model decisions with expected labels."""

    expected_ids = {
        case["evidence_id"]
        for case in TEST_CASES
    }

    returned_ids = [
        decision.evidence_id
        for decision in decisions
    ]

    if (
        len(returned_ids) != len(expected_ids)
        or set(returned_ids) != expected_ids
    ):
        raise ValueError(
            "The verifier must return exactly one "
            "decision for every test case."
        )

    decision_lookup = {
        decision.evidence_id: decision
        for decision in decisions
    }

    results = []

    correct = 0
    false_acceptances = 0
    negative_count = 0

    for case in TEST_CASES:
        decision = decision_lookup[
            case["evidence_id"]
        ]

        predicted = decision.status
        expected = case["expected_status"]

        passed = predicted == expected

        if passed:
            correct += 1

        if expected == "unsupported":
            negative_count += 1

            if predicted == "supported":
                false_acceptances += 1

        results.append({
            "evidence_id": case["evidence_id"],
            "field": case["field"],
            "text": case["text"],
            "expected_status": expected,
            "predicted_status": predicted,
            "passed": passed,
            "explanation": decision.explanation,
        })

    accuracy = correct / len(TEST_CASES)

    false_acceptance_rate = (
        false_acceptances / negative_count
        if negative_count
        else 0.0
    )

    return {
        "total_tests": len(TEST_CASES),
        "correct": correct,
        "incorrect": len(TEST_CASES) - correct,
        "accuracy": accuracy,
        "false_acceptances": false_acceptances,
        "negative_tests": negative_count,
        "false_acceptance_rate": false_acceptance_rate,
        "all_tests_passed": (
            correct == len(TEST_CASES)
        ),
        "results": results,
    }


def main():
    print(
        "=== LESSON 6B.6: "
        "VISUAL VERIFICATION NEGATIVE TEST ==="
    )

    load_dotenv()

    if not PDF_PATH.is_file():
        raise FileNotFoundError(
            f"Missing scanned PDF: {PDF_PATH}"
        )

    print("Source PDF:", PDF_PATH)
    print("Model:", MODEL)
    print("Test cases:", len(TEST_CASES))

    positive_count = sum(
        case["expected_status"] == "supported"
        for case in TEST_CASES
    )

    negative_count = sum(
        case["expected_status"] == "unsupported"
        for case in TEST_CASES
    )

    print("Positive tests:", positive_count)
    print("Negative tests:", negative_count)

    image_bytes = render_page()

    print(
        "Rendered image bytes:",
        len(image_bytes),
    )

    # The expected labels are deliberately NOT sent
    # to the model. They are only used afterward
    # to evaluate the returned decisions.
    model_evidence = [
        {
            "evidence_id": case["evidence_id"],
            "field": case["field"],
            "text": case["text"],
        }
        for case in TEST_CASES
    ]

    print(
        "\nSending one visual verification request..."
    )

    client = OpenAI()

    verification = verify_with_vision(
        client,
        image_bytes,
        model_evidence,
    )

    evaluation = evaluate_decisions(
        verification.decisions
    )

    print("\nINDIVIDUAL TEST RESULTS")

    for result in evaluation["results"]:
        status = (
            "PASS" if result["passed"] else "FAIL"
        )

        print(
            f"\n{result['evidence_id']}: {status}"
        )

        print(
            "Evidence:",
            result["text"],
        )

        print(
            "Expected:",
            result["expected_status"],
        )

        print(
            "Predicted:",
            result["predicted_status"],
        )

        print(
            "Reason:",
            result["explanation"],
        )

    print("\nEVALUATION SUMMARY")

    print(
        "Correct:",
        f"{evaluation['correct']}/"
        f"{evaluation['total_tests']}",
    )

    print(
        "Accuracy:",
        f"{evaluation['accuracy']:.3f}",
    )

    print(
        "False acceptances:",
        evaluation["false_acceptances"],
    )

    print(
        "False acceptance rate:",
        f"{evaluation['false_acceptance_rate']:.3f}",
    )

    print(
        "All tests passed:",
        evaluation["all_tests_passed"],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "source_file": str(PDF_PATH),
                "model": MODEL,
                "evaluation": evaluation,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT_PATH)

    if not evaluation["all_tests_passed"]:
        raise SystemExit(
            "\n=== NEGATIVE TEST FAILED ==="
        )

    print(
        "\n=== LESSON 6B.6 PASSED ==="
    )


if __name__ == "__main__":
    main()
