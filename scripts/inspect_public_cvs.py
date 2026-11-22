
import json
from collections import Counter
from pathlib import Path


DATA_PATH = Path("data/public_benchmark/resumes.jsonl")


def main():
    with DATA_PATH.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not records:
        raise SystemExit("No CV records found.")

    roles = Counter(
        record["desired_position"]
        for record in records
    )

    print("=== DATASET OVERVIEW ===")
    print("Total CVs:", len(records))
    print("Unique roles:", len(roles))

    print("\n=== TOP 10 ROLES ===")
    for role, count in roles.most_common(10):
        print(f"{role}: {count}")

    example = records[0]
    truth = example["ground_truth"]

    print("\n=== FIRST CANDIDATE ===")
    print("Sample ID:", example["sample_id"])
    print("Desired position:", example["desired_position"])
    print("Years of experience:", example["years_experience"])

    for field in [
        "skills",
        "experiences",
        "projects",
        "educations",
        "certificates",
    ]:
        value = truth.get(field)

        print(f"\n=== {field.upper()} ===")
        print("Python type:", type(value).__name__)

        if isinstance(value, list):
            print("Number of entries:", len(value))
            print(
                json.dumps(
                    value[:2],
                    indent=2,
                    ensure_ascii=False,
                )[:1500]
            )
        else:
            print(
                json.dumps(
                    value,
                    indent=2,
                    ensure_ascii=False,
                )[:1500]
            )


if __name__ == "__main__":
    main()
