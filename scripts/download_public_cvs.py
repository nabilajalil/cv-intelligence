
import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASET_NAME = "sukhrobnurali/resume-parsing-vision"
OUTPUT_DIR = Path("data/public_benchmark")


def download_resumes(limit: int = 50) -> Path:
    """Download text and ground truth from public synthetic CVs."""

    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    print(f"Loading dataset: {DATASET_NAME}")
    print("Split: train")
    print(f"Requested samples: {limit}")

    # Load only the required training rows.
    # Decode images only when we need them in the multimodal lesson.
    dataset = load_dataset(
        DATASET_NAME,
        split=f"train[:{limit}]",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "resumes.jsonl"

    valid_records = []

    for row in dataset:
        resume_text = row.get("resume_text")
        ground_truth = row.get("ground_truth")

        if not isinstance(resume_text, str) or not resume_text.strip():
            continue

        if isinstance(ground_truth, str):
            ground_truth = json.loads(ground_truth)

        if not isinstance(ground_truth, dict):
            continue

        record = {
            "sample_id": row["sample_id"],
            "resume_text": resume_text,
            "ground_truth": ground_truth,
            "desired_position": row.get("desired_position"),
            "years_experience": row.get("years_experience"),
            "num_pages": row.get("num_pages"),
            "template_id": row.get("template_id"),
            "theme_id": row.get("theme_id"),
            "split": row.get("split", "train"),
        }

        valid_records.append(record)

    with output_path.open("w", encoding="utf-8") as file:
        for record in valid_records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    print("\n=== DOWNLOAD COMPLETE ===")
    print("Saved records:", len(valid_records))
    print("Output file:", output_path)

    if valid_records:
        example = valid_records[0]

        print("\n=== FIRST SAMPLE ===")
        print("Sample ID:", example["sample_id"])
        print("Position:", example["desired_position"])
        print("Experience:", example["years_experience"])
        print("Pages:", example["num_pages"])

        print("\n=== RESUME TEXT PREVIEW ===")
        print(example["resume_text"][:500])

        print("\n=== GROUND TRUTH FIELDS ===")
        print(list(example["ground_truth"].keys()))

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    download_resumes(limit=args.limit)


if __name__ == "__main__":
    main()
