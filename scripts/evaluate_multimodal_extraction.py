
"""Lesson 6A.3 — Evaluate vision-based CV extraction.

Compares a saved vision prediction against the known
synthetic CV created in Lesson 6A.1.

No OpenAI API calls are made.

Run:
    python -m scripts.evaluate_multimodal_extraction
"""

import json
import re
from pathlib import Path


PREDICTION_PATH = Path(
    "data/multimodal/reports/"
    "synthetic_scanned_cv_vision.json"
)

OUTPUT_PATH = Path(
    "data/multimodal/reports/"
    "synthetic_scanned_cv_evaluation.json"
)


# Ground truth from create_multimodal_test_pdf.py.
# This fixture is specific to our synthetic M001 CV.
GROUND_TRUTH = {
    "candidate_id": "M001",
    "role": "Software Engineer",
    "skills": [
        "Python",
        "SQL",
        "Docker",
        "GitLab",
    ],
    "experience": [
        "Built a Python data-processing pipeline.",
        "Used SQL for reporting and analysis.",
        "Containerized an application with Docker.",
    ],
    "projects": [
        "Recruitment Analytics Dashboard",
        "Developed a dashboard for synthetic data.",
    ],
    "education": [
        "Bachelor of Computer Science",
    ],
}


FIELDS = [
    "candidate_id",
    "role",
    "skills",
    "experience",
    "projects",
    "education",
]

LIST_FIELDS = [
    "skills",
    "experience",
    "projects",
    "education",
]


def normalize(value):
    """Normalize text for exact-match comparisons."""

    if not isinstance(value, str):
        return ""

    value = value.casefold().strip()

    # Normalize whitespace and punctuation.
    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalized_set(values):
    """Convert a list of strings to a normalized set."""

    return {
        normalize(value)
        for value in values
        if normalize(value)
    }


def precision_recall_f1(expected, predicted):
    """Calculate set-based precision, recall, and F1."""

    expected_set = normalized_set(expected)
    predicted_set = normalized_set(predicted)

    correct = expected_set & predicted_set

    precision = (
        len(correct) / len(predicted_set)
        if predicted_set
        else 0.0
    )

    recall = (
        len(correct) / len(expected_set)
        if expected_set
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "expected_count": len(expected_set),
        "predicted_count": len(predicted_set),
        "correct_count": len(correct),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "missing": sorted(
            expected_set - predicted_set
        ),
        "unsupported": sorted(
            predicted_set - expected_set
        ),
    }


def field_completeness(prediction):
    """Measure presence of the six expected fields."""

    present = 0

    for field in FIELDS:
        value = prediction.get(field)

        if isinstance(value, str):
            has_value = bool(value.strip())
        elif isinstance(value, list):
            has_value = bool(value)
        else:
            has_value = False

        if has_value:
            present += 1

    return {
        "present_fields": present,
        "total_fields": len(FIELDS),
        "score": present / len(FIELDS),
    }


def evaluate(prediction):
    """Evaluate one structured CV prediction."""

    results = {
        "candidate_id_match": (
            normalize(
                prediction.get("candidate_id")
            )
            == normalize(
                GROUND_TRUTH["candidate_id"]
            )
        ),
        "role_match": (
            normalize(
                prediction.get("role")
            )
            == normalize(
                GROUND_TRUTH["role"]
            )
        ),
        "field_completeness": (
            field_completeness(prediction)
        ),
        "list_metrics": {},
    }

    for field in LIST_FIELDS:
        predicted_values = prediction.get(
            field,
            [],
        )

        if not isinstance(predicted_values, list):
            predicted_values = []

        results["list_metrics"][field] = (
            precision_recall_f1(
                GROUND_TRUTH[field],
                predicted_values,
            )
        )

    all_scalar_matches = [
        results["candidate_id_match"],
        results["role_match"],
    ]

    all_list_matches = [
        (
            results["list_metrics"][field]["f1"]
            == 1.0
        )
        for field in LIST_FIELDS
    ]

    results["all_fields_match"] = all(
        all_scalar_matches + all_list_matches
    )

    return results


def main():
    print(
        "=== LESSON 6A.3: "
        "MULTIMODAL EXTRACTION EVALUATION ==="
    )

    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Missing prediction: {PREDICTION_PATH}"
        )

    prediction = json.loads(
        PREDICTION_PATH.read_text(
            encoding="utf-8"
        )
    )

    results = evaluate(prediction)

    print("\nSCALAR FIELDS")
    print(
        "Candidate ID match:",
        results["candidate_id_match"],
    )
    print(
        "Role match:",
        results["role_match"],
    )

    completeness = results[
        "field_completeness"
    ]

    print("\nFIELD COMPLETENESS")
    print(
        f"{completeness['present_fields']}/"
        f"{completeness['total_fields']}"
    )
    print(
        "Score:",
        f"{completeness['score']:.3f}",
    )

    print("\nLIST EXTRACTION METRICS")

    for field in LIST_FIELDS:
        metrics = results[
            "list_metrics"
        ][field]

        print(f"\n{field.upper()}")
        print(
            "Correct:",
            f"{metrics['correct_count']}/"
            f"{metrics['expected_count']}",
        )
        print(
            "Precision:",
            f"{metrics['precision']:.3f}",
        )
        print(
            "Recall:",
            f"{metrics['recall']:.3f}",
        )
        print(
            "F1:",
            f"{metrics['f1']:.3f}",
        )
        print(
            "Missing:",
            metrics["missing"],
        )
        print(
            "Unsupported:",
            metrics["unsupported"],
        )

    print("\nFINAL RESULT")
    print(
        "All fields match:",
        results["all_fields_match"],
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "prediction_path": str(
                    PREDICTION_PATH
                ),
                "ground_truth": GROUND_TRUTH,
                "evaluation": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT_PATH)

    print(
        "\n=== EVALUATION COMPLETE ==="
    )


if __name__ == "__main__":
    main()
