
"""Lesson 6A.5 — Hybrid extraction consistency.

Compares text-route and vision-route CV profiles.

No OpenAI API calls.

Run:
    python -m scripts.evaluation.evaluate_hybrid_consistency
"""

import json
import re
from pathlib import Path


BASE_DIR = Path("data/multimodal")

TEXT_PATH = (
    BASE_DIR
    / "hybrid_profiles/synthetic_text_cv_profile.json"
)

VISION_PATH = (
    BASE_DIR
    / "hybrid_profiles/synthetic_scanned_cv_profile.json"
)

OUTPUT_PATH = (
    BASE_DIR
    / "reports/hybrid_consistency_evaluation.json"
)

FIELDS = [
    "candidate_id",
    "role",
    "skills",
    "experience",
    "projects",
    "education",
]

LIST_FIELDS = {
    "skills",
    "experience",
    "projects",
    "education",
}


def normalize(value):
    """Normalize a string for comparison."""

    if not isinstance(value, str):
        return ""

    value = value.casefold().strip()

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def normalize_list(values):
    """Compare list content regardless of ordering."""

    if not isinstance(values, list):
        return []

    return sorted({
        normalize(value)
        for value in values
        if normalize(value)
    })


def compare_profiles(text_profile, vision_profile):
    """Compare the six structured CV fields."""

    results = {}
    matching_fields = 0

    for field in FIELDS:
        text_value = text_profile.get(field)
        vision_value = vision_profile.get(field)

        if field in LIST_FIELDS:
            text_normalized = normalize_list(text_value)
            vision_normalized = normalize_list(vision_value)
        else:
            text_normalized = normalize(text_value)
            vision_normalized = normalize(vision_value)

        matches = text_normalized == vision_normalized

        if matches:
            matching_fields += 1

        results[field] = {
            "match": matches,
            "text_value": text_value,
            "vision_value": vision_value,
        }

    return {
        "matching_fields": matching_fields,
        "total_fields": len(FIELDS),
        "consistency_score": (
            matching_fields / len(FIELDS)
        ),
        "all_fields_match": (
            matching_fields == len(FIELDS)
        ),
        "field_results": results,
    }


def main():
    print(
        "=== LESSON 6A.5: "
        "HYBRID EXTRACTION CONSISTENCY ==="
    )

    for path in [TEXT_PATH, VISION_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    text_result = json.loads(
        TEXT_PATH.read_text(encoding="utf-8")
    )

    vision_result = json.loads(
        VISION_PATH.read_text(encoding="utf-8")
    )

    print("\nROUTING CHECK")

    text_method = text_result.get(
        "extraction_method"
    )

    vision_method = vision_result.get(
        "extraction_method"
    )

    routing_correct = (
        text_method == "text"
        and vision_method == "vision"
    )

    print("Text PDF method:", text_method)
    print("Scanned PDF method:", vision_method)
    print("Routing correct:", routing_correct)

    comparison = compare_profiles(
        text_result["profile"],
        vision_result["profile"],
    )

    print("\nFIELD COMPARISON")

    for field, result in comparison[
        "field_results"
    ].items():
        status = (
            "PASS" if result["match"] else "FAIL"
        )

        print(
            f"{field:15s}: {status}"
        )

    print("\nFINAL RESULTS")
    print(
        "Matching fields:",
        f"{comparison['matching_fields']}/"
        f"{comparison['total_fields']}",
    )
    print(
        "Consistency score:",
        f"{comparison['consistency_score']:.3f}",
    )
    print(
        "All fields match:",
        comparison["all_fields_match"],
    )

    evaluation = {
        "text_file": str(TEXT_PATH),
        "vision_file": str(VISION_PATH),
        "routing_correct": routing_correct,
        "comparison": comparison,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            evaluation,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT_PATH)

    if not routing_correct:
        raise SystemExit(
            "FAIL: Incorrect extraction routing."
        )

    if not comparison["all_fields_match"]:
        raise SystemExit(
            "FAIL: Extraction profiles differ."
        )

    print(
        "\n=== LESSON 6A.5 PASSED ==="
    )


if __name__ == "__main__":
    main()
