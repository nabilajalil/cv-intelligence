
import csv
import json
from pathlib import Path


DATA_DIR = Path("data/public_benchmark")

QUERY_SPECS = [
    {
        "query_id": "q001",
        "text": "Find candidates with Python and Docker",
        "required_skills": ["Python", "Docker"],
    },
    {
        "query_id": "q002",
        "text": "Find candidates with Python and SQL",
        "required_skills": ["Python", "Sql"],
    },
    {
        "query_id": "q003",
        "text": "Find candidates with Java and Linux",
        "required_skills": ["Java", "Linux"],
    },
    {
        "query_id": "q004",
        "text": "Find candidates with Docker and GitLab",
        "required_skills": ["Docker", "Gitlab"],
    },
    {
        "query_id": "q005",
        "text": "Find candidates with Python and JavaScript",
        "required_skills": ["Python", "JavaScript"],
    },
]


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def extract_skills(record: dict) -> set[str]:
    """Read canonical skills from the structured ground truth."""
    skills = record["ground_truth"].get("skills") or []

    return {
        item["skill_name"].strip().casefold()
        for item in skills
        if isinstance(item, dict)
        and isinstance(item.get("skill_name"), str)
        and item["skill_name"].strip()
    }


def relevance_grade(
    record: dict,
    required_skills: list[str],
) -> int:
    """2 = all required skills, 1 = some, 0 = none."""
    candidate_skills = extract_skills(record)

    matches = sum(
        skill.casefold() in candidate_skills
        for skill in required_skills
    )

    if matches == len(required_skills):
        return 2
    if matches > 0:
        return 1
    return 0


def build_benchmark(
    data_dir: Path = DATA_DIR,
    query_specs: list[dict] = QUERY_SPECS,
) -> tuple[Path, Path]:
    records = load_records(data_dir / "resumes.jsonl")

    queries_path = data_dir / "queries.jsonl"
    qrels_path = data_dir / "qrels.tsv"

    with queries_path.open("w", encoding="utf-8") as file:
        for query in query_specs:
            file.write(json.dumps(query) + "\n")

    with qrels_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["query_id", "candidate_id", "relevance"])

        for query in query_specs:
            for record in records:
                grade = relevance_grade(
                    record,
                    query["required_skills"],
                )

                writer.writerow([
                    query["query_id"],
                    record["sample_id"],
                    grade,
                ])

    return queries_path, qrels_path


def main():
    queries_path, qrels_path = build_benchmark()

    print("=== PUBLIC CV BENCHMARK ===")
    print("Queries:", queries_path)
    print("Relevance judgments:", qrels_path)

    records = load_records(DATA_DIR / "resumes.jsonl")

    for query in QUERY_SPECS:
        grades = [
            relevance_grade(record, query["required_skills"])
            for record in records
        ]

        print(f"\n{query['query_id']}: {query['text']}")
        print("Full matches:", grades.count(2))
        print("Partial matches:", grades.count(1))
        print("No matches:", grades.count(0))


if __name__ == "__main__":
    main()
