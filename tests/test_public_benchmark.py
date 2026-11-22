
from scripts.build_public_benchmark import (
    extract_skills,
    relevance_grade,
)


def make_record(skills):
    return {
        "ground_truth": {
            "skills": [
                {"skill_name": skill, "level": 3}
                for skill in skills
            ]
        }
    }


def test_extract_skills():
    record = make_record(["Python", "Docker"])

    assert extract_skills(record) == {"python", "docker"}


def test_full_skill_match():
    record = make_record(["Python", "Docker", "AWS"])

    assert relevance_grade(
        record, ["Python", "Docker"]
    ) == 2


def test_partial_skill_match():
    record = make_record(["Python", "SQL"])

    assert relevance_grade(
        record, ["Python", "Docker"]
    ) == 1


def test_no_skill_match():
    record = make_record(["Java", "Linux"])

    assert relevance_grade(
        record, ["Python", "Docker"]
    ) == 0
