
from cv_intelligence.candidate_summary import summarize_candidate


def test_summary_with_matching_skills():
    candidate = {
        "matched_skills": ["Python", "Docker"],
        "skills_not_found_in_evidence": ["AWS"],
        "skill_evidence": {
            "Python": [
                {
                    "section": "skills",
                    "page": 1,
                    "text": "Python, Docker",
                }
            ],
            "Docker": [
                {
                    "section": "projects",
                    "page": 1,
                    "text": "Built an application using Docker.",
                }
            ],
        },
    }

    summary = summarize_candidate(candidate)

    assert "Python" in summary
    assert "Docker" in summary
    assert "AWS" in summary
    assert "skills, projects" not in summary
    assert "projects, skills" in summary
    assert "not independently verified" in summary


def test_summary_without_matching_skills():
    candidate = {
        "matched_skills": [],
        "skills_not_found_in_evidence": ["Kubernetes"],
        "skill_evidence": {},
    }

    summary = summarize_candidate(candidate)

    assert "No requested technical skills" in summary
    assert "Kubernetes" in summary

