
"""Lesson 4G: Structured evidence-quality verification.

For the top five BM25 candidates:
1. Gather all indexed chunks belonging to each candidate.
2. Check explicit mentions of every required skill.
3. Ask an LLM to classify the strength of the evidence.
4. Validate its chunk IDs and quoted excerpts.
5. Print and save an auditable evidence report.

This is an experimental script. It does not change the
existing LangGraph agent or benchmark scripts.
"""

import json
import re
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from scripts.retrieval.demo_parent_cv_verification import (
    collect_candidate_chunks,
    extract_required_skills,
    find_skill_evidence,
    load_chunks,
)

MODEL = "gpt-4.1-mini"
TOP_K = 5

OUTPUT_PATH = Path(
    "data/public_benchmark/evidence_quality_results.json"
)


class SkillAssessment(BaseModel):
    skill: str = Field(
        description="One required technology, copied exactly."
    )

    level: Literal[
        "professional_use",
        "project_use",
        "skill_listing",
        "learning_only",
        "no_evidence",
        "unclear",
    ]

    chunk_id: str | None = Field(
        description="ID of the chunk supporting this assessment."
    )

    quote: str | None = Field(
        description="Short verbatim excerpt from the cited chunk."
    )

    explanation: str = Field(
        description="Brief explanation of the classification."
    )


class CandidateAssessment(BaseModel):
    assessments: list[SkillAssessment]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rank_candidates(chunks, query, limit=TOP_K):
    """Rank unique candidates by their strongest BM25 chunk."""

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    scores = bm25.get_scores(tokenize(query))

    best_scores = {}

    for chunk, score in zip(chunks, scores):
        candidate_id = chunk["candidate_id"]
        score = float(score)

        if (
            candidate_id not in best_scores
            or score > best_scores[candidate_id]
        ):
            best_scores[candidate_id] = score

    ranking = sorted(
        best_scores,
        key=lambda candidate_id: (
            -best_scores[candidate_id],
            candidate_id,
        ),
    )

    return ranking[:limit]


def normalize_whitespace(value: str) -> str:
    return " ".join(value.split())


def validate_assessment(
    assessment: SkillAssessment,
    expected_skill: str,
    chunks: list[dict],
) -> dict:
    """Check that the citation points to actual CV evidence.

    A valid quote must occur in the referenced chunk.
    The quote must also explicitly contain the assessed skill.

    This checks citation integrity, not whether the LLM's
    interpretation of the excerpt is objectively correct.
    """

    result = assessment.model_dump()
    result["citation_valid"] = False
    result["validation_issue"] = None

    if assessment.skill != expected_skill:
        result["validation_issue"] = "Skill name mismatch"
        return result

    if assessment.level in ("no_evidence", "unclear"):
        # These labels may legitimately have no supporting quote.
        if assessment.chunk_id is None and assessment.quote is None:
            result["citation_valid"] = True
            return result

        result["validation_issue"] = (
            "Unclear/no-evidence assessment should not cite "
            "a positive skill excerpt"
        )
        return result

    if not assessment.chunk_id or not assessment.quote:
        result["validation_issue"] = "Missing chunk ID or quote"
        return result

    chunk_lookup = {
        chunk["chunk_id"]: chunk["text"]
        for chunk in chunks
    }

    chunk_text = chunk_lookup.get(assessment.chunk_id)

    if chunk_text is None:
        result["validation_issue"] = "Unknown chunk ID"
        return result

    normalized_quote = normalize_whitespace(assessment.quote)
    normalized_chunk = normalize_whitespace(chunk_text)

    if not normalized_quote:
        result["validation_issue"] = "Empty quote"
        return result

    if normalized_quote not in normalized_chunk:
        result["validation_issue"] = (
            "Quote not found verbatim in cited chunk"
        )
        return result

    skill_pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(expected_skill)}"
        rf"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )

    if not skill_pattern.search(normalized_quote):
        result["validation_issue"] = (
            "Quote does not explicitly mention the skill"
        )
        return result

    result["citation_valid"] = True
    return result


def grade_candidate(
    llm,
    candidate_id: str,
    candidate_chunks: list[dict],
    required_skills: list[str],
) -> dict:
    """Grade one candidate using their indexed CV evidence."""

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in candidate_chunks
    }

    # Only include chunks mentioning at least one required
    # skill. This reduces the prompt size while retaining
    # all directly matching evidence chunks.
    mention_evidence = find_skill_evidence(
        candidate_chunks,
        required_skills,
    )

    relevant_chunk_ids = set()

    for ids in mention_evidence.values():
        relevant_chunk_ids.update(ids)

    relevant_chunks = [
        chunk
        for chunk in candidate_chunks
        if chunk["chunk_id"] in relevant_chunk_ids
    ]

    # Include all indexed chunks as additional context.
    # The model may need surrounding information to tell
    # whether a mention is from employment, education,
    # project work, or a general skills list.
    cv_context = "\n\n".join(
        f"CHUNK ID: {chunk['chunk_id']}\n"
        f"TEXT:\n{chunk['text']}"
        for chunk in candidate_chunks
    )

    system_prompt = """
You are a cautious CV evidence analyst.

Evaluate each required technology independently using
ONLY the supplied CV chunks.

Use these classifications:

professional_use:
  The CV explicitly connects the technology to work
  performed in a professional/employment context.

project_use:
  The CV explicitly connects the technology to a specific
  project, but professional employment use is not established.

skill_listing:
  The technology appears in a skills list or tech stack,
  without clear evidence of its actual use.

learning_only:
  The CV only describes learning, studying, or intending
  to learn the technology.

no_evidence:
  The technology does not appear explicitly in the CV.

unclear:
  The text is ambiguous, contradictory, or insufficient
  for a confident classification.

Important:
- Do not assume that listing a skill proves proficiency.
- Do not assume that a project was professional employment.
- Do not infer years of experience.
- A job title alone does not establish technology usage.
- Do not invent projects, employers, or experience.
- Return exactly one assessment per required skill.
- For a positive evidence category, provide ONE chunk ID
  and a short EXACT quotation from that chunk.
- The quote must explicitly contain the required skill.
- Do not paraphrase inside the quote.
- For no_evidence or unclear, use null for both citation
  fields.
- If evidence supports multiple categories, prefer the
  strongest category directly justified by the text.
"""

    structured_llm = llm.with_structured_output(
        CandidateAssessment,
        method="json_schema",
    )

    response = structured_llm.invoke([
        ("system", system_prompt),
        (
            "human",
            f"Candidate ID: {candidate_id}\n"
            f"Required skills: "
            f"{json.dumps(required_skills)}\n\n"
            f"CV evidence:\n{cv_context}"
        ),
    ])

    # Avoid silently accepting missing or duplicate skills.
    response_by_skill = {}

    for item in response.assessments:
        response_by_skill.setdefault(
            item.skill,
            [],
        ).append(item)

    validated = []

    for skill in required_skills:
        items = response_by_skill.get(skill, [])

        if len(items) != 1:
            validated.append({
                "skill": skill,
                "level": "unclear",
                "chunk_id": None,
                "quote": None,
                "explanation": (
                    "The model did not return exactly one "
                    "assessment for this skill."
                ),
                "citation_valid": False,
                "validation_issue": (
                    "Missing or duplicate skill assessment"
                ),
            })
            continue

        validated.append(
            validate_assessment(
                items[0],
                skill,
                candidate_chunks,
            )
        )

    return {
        "candidate_id": candidate_id,
        "required_skills": required_skills,
        "explicit_mentions": mention_evidence,
        "evidence_chunks": [
            chunk["chunk_id"]
            for chunk in relevant_chunks
        ],
        "assessments": validated,
    }


def print_candidate_report(report: dict):
    print(f"\nCandidate: {report['candidate_id']}")

    for assessment in report["assessments"]:
        print(f"\n  Skill: {assessment['skill']}")
        print(f"  Level: {assessment['level']}")

        print(
            "  Citation valid:",
            assessment["citation_valid"],
        )

        if assessment["chunk_id"]:
            print(
                "  Chunk:",
                assessment["chunk_id"],
            )

        if assessment["quote"]:
            print(
                "  Quote:",
                assessment["quote"],
            )

        print(
            "  Explanation:",
            assessment["explanation"],
        )

        if assessment["validation_issue"]:
            print(
                "  VALIDATION WARNING:",
                assessment["validation_issue"],
            )


def main():
    load_dotenv()

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Find candidates with Python and SQL"
    )

    required_skills = extract_required_skills(query)

    print("=== LESSON 4G: EVIDENCE QUALITY ===")
    print("Query:", query)
    print("Required skills:", required_skills)

    if not required_skills:
        print(
            "No supported benchmark technology was found. "
            "Try: Find candidates with Python and SQL"
        )
        return

    chunks = load_chunks()
    parents = collect_candidate_chunks(chunks)

    candidate_ids = rank_candidates(
        chunks,
        query,
        limit=TOP_K,
    )

    print("Indexed chunks:", len(chunks))
    print("Candidates:", len(parents))
    print("Top candidates:", candidate_ids)

    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
    )

    reports = []

    for candidate_id in candidate_ids:
        print(
            f"\n[GRADE] Evaluating {candidate_id} "
            f"across {len(parents[candidate_id])} chunks..."
        )

        try:
            report = grade_candidate(
                llm,
                candidate_id,
                parents[candidate_id],
                required_skills,
            )

            reports.append(report)
            print_candidate_report(report)

        except Exception as error:
            print(
                f"[ERROR] Could not grade {candidate_id}: "
                f"{type(error).__name__}: {error}"
            )

            reports.append({
                "candidate_id": candidate_id,
                "error": str(error),
                "assessments": [],
            })

    output = {
        "lesson": "4G",
        "query": query,
        "required_skills": required_skills,
        "model": MODEL,
        "candidate_reports": reports,
        "limitations": [
            "Only explicit benchmark technologies are extracted.",
            "The LLM's evidence classification may be wrong.",
            "Citation validation checks quoted text, not truth.",
            "A CV is a self-reported source.",
            "Results are not independently benchmarked.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")

    total = sum(
        len(report["assessments"])
        for report in reports
    )

    valid = sum(
        assessment["citation_valid"]
        for report in reports
        for assessment in report["assessments"]
    )

    print("Skill assessments:", total)
    print("Valid citations:", valid)
    print("Invalid citations:", total - valid)
    print("Saved report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
