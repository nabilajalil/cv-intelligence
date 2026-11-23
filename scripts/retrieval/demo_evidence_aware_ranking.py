
"""Lesson 4H: Evidence-aware candidate ranking.

Pipeline:
    BM25 ranking
    -> parent CV skill verification
    -> structured LLM evidence assessment
    -> validate skill and context quotes
    -> conservative evidence-based reranking

This is an experiment, not a production hiring decision system.
Existing scripts remain unchanged.
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

# Limit LLM costs while testing.
RETRIEVAL_POOL = 10
TOP_K = 5

OUTPUT_PATH = Path(
    "data/public_benchmark/evidence_aware_ranking.json"
)

LEVEL_SCORES = {
    "professional_use": 4,
    "project_use": 3,
    "skill_listing": 2,
    "learning_only": 1,
    "unclear": 0,
    "no_evidence": 0,
}


class SkillEvidence(BaseModel):
    skill: str

    level: Literal[
        "professional_use",
        "project_use",
        "skill_listing",
        "learning_only",
        "unclear",
        "no_evidence",
    ]

    skill_chunk_id: str | None = None
    skill_quote: str | None = None

    context_chunk_id: str | None = None
    context_quote: str | None = None

    explanation: str = Field(
        description="Why this evidence level was selected."
    )


class CandidateEvidence(BaseModel):
    assessments: list[SkillEvidence]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def skill_pattern(skill: str):
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(skill)}"
        rf"(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


def quote_exists(
    quote: str | None,
    chunk_id: str | None,
    chunks: list[dict],
) -> bool:
    """Verify that a quotation occurs in its cited chunk."""

    if not quote or not chunk_id:
        return False

    chunk_lookup = {
        chunk["chunk_id"]: chunk["text"]
        for chunk in chunks
    }

    source_text = chunk_lookup.get(chunk_id)

    if source_text is None:
        return False

    return (
        normalize_text(quote)
        in normalize_text(source_text)
    )


def validate_skill(
    assessment: SkillEvidence,
    required_skill: str,
    chunks: list[dict],
) -> dict:
    """Validate citations and apply conservative downgrades."""

    result = assessment.model_dump()
    original_level = assessment.level

    result["original_level"] = original_level
    result["validated_level"] = original_level
    result["warnings"] = []

    if assessment.skill != required_skill:
        result["validated_level"] = "unclear"
        result["warnings"].append(
            "Returned skill does not match required skill."
        )
        return result

    valid_skill_quote = quote_exists(
        assessment.skill_quote,
        assessment.skill_chunk_id,
        chunks,
    )

    if valid_skill_quote:
        valid_skill_quote = bool(
            skill_pattern(required_skill).search(
                assessment.skill_quote
            )
        )

    valid_context_quote = quote_exists(
        assessment.context_quote,
        assessment.context_chunk_id,
        chunks,
    )

    result["skill_quote_valid"] = valid_skill_quote
    result["context_quote_valid"] = valid_context_quote

    # Positive evidence requires an exact skill mention.
    if original_level in (
        "professional_use",
        "project_use",
        "skill_listing",
        "learning_only",
    ):
        if not valid_skill_quote:
            result["validated_level"] = "unclear"
            result["warnings"].append(
                "Missing or invalid skill quotation."
            )
            return result

    # Stronger classifications require context evidence.
    if original_level in (
        "professional_use",
        "project_use",
    ):
        if not valid_context_quote:
            result["validated_level"] = "skill_listing"
            result["warnings"].append(
                "No valid context quotation. "
                "Downgraded to skill_listing."
            )

    result["score"] = LEVEL_SCORES[
        result["validated_level"]
    ]

    return result


def rank_bm25(chunks, query):
    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    scores = bm25.get_scores(tokenize(query))

    best = {}

    for chunk, score in zip(chunks, scores):
        candidate_id = chunk["candidate_id"]
        score = float(score)

        if (
            candidate_id not in best
            or score > best[candidate_id]
        ):
            best[candidate_id] = score

    ranked = sorted(
        best,
        key=lambda candidate_id: (
            -best[candidate_id],
            candidate_id,
        ),
    )

    return ranked, best


def assess_candidate(
    llm,
    candidate_id,
    candidate_chunks,
    required_skills,
):
    """Ask the LLM to identify skill and context evidence."""

    cv_text = "\n\n".join(
        f"CHUNK ID: {chunk['chunk_id']}\n"
        f"TEXT:\n{chunk['text']}"
        for chunk in candidate_chunks
    )

    structured_llm = llm.with_structured_output(
        CandidateEvidence,
        method="json_schema",
    )

    response = structured_llm.invoke([
        (
            "system",
            """
You are a cautious CV evidence analyst.

Classify each required technology independently.

Categories:
- professional_use: explicitly connected to specific
  professional employment work.
- project_use: explicitly connected to a specific project,
  but employment context is not established.
- skill_listing: explicitly mentioned, but actual use
  is not established.
- learning_only: only described as learning or studying.
- no_evidence: no explicit technology mention.
- unclear: ambiguous or contradictory evidence.

For EACH skill return:
1. The exact skill name.
2. One evidence level.
3. A short VERBATIM skill quotation and its chunk ID.
4. For professional_use or project_use, a separate
   VERBATIM context quotation and its chunk ID.
5. A brief explanation.

Rules:
- Use only supplied CV chunks.
- Skill quotes must explicitly contain the technology.
- A generic skills list is not professional-use proof.
- A 'Tech:' list alone is not enough for professional_use.
- Context must describe a specific project or work activity.
- The context quotation may be from a different chunk.
- Never invent experience, employers, or project details.
- If context is insufficient, select skill_listing.
- For no_evidence or unclear, use null quotation fields.
- Return exactly one assessment per required skill.
"""
        ),
        (
            "human",
            f"Candidate: {candidate_id}\n"
            f"Required skills: "
            f"{json.dumps(required_skills)}\n\n"
            f"CV:\n{cv_text}"
        ),
    ])

    by_skill = {}

    for item in response.assessments:
        by_skill.setdefault(item.skill, []).append(item)

    assessments = []

    for skill in required_skills:
        matches = by_skill.get(skill, [])

        if len(matches) != 1:
            assessments.append({
                "skill": skill,
                "original_level": "unclear",
                "validated_level": "unclear",
                "score": 0,
                "skill_quote_valid": False,
                "context_quote_valid": False,
                "warnings": [
                    "Missing or duplicate LLM assessment."
                ],
            })
            continue

        validated = validate_skill(
            matches[0],
            skill,
            candidate_chunks,
        )

        validated["score"] = LEVEL_SCORES[
            validated["validated_level"]
        ]

        assessments.append(validated)

    return assessments


def main():
    load_dotenv()

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Find candidates with Python and SQL"
    )

    required_skills = extract_required_skills(query)

    print("=== LESSON 4H: EVIDENCE-AWARE RANKING ===")
    print("Query:", query)
    print("Required skills:", required_skills)

    if not required_skills:
        print("No supported technology requirements found.")
        return

    chunks = load_chunks()
    parents = collect_candidate_chunks(chunks)

    bm25_ranked, bm25_scores = rank_bm25(
        chunks,
        query,
    )

    # Verify explicit skill mentions before spending
    # money on LLM assessment.
    verified = []

    for candidate_id in bm25_ranked:
        evidence = find_skill_evidence(
            parents[candidate_id],
            required_skills,
        )

        if all(evidence[skill] for skill in required_skills):
            verified.append(candidate_id)

    selected = verified[:RETRIEVAL_POOL]

    print("Indexed chunks:", len(chunks))
    print("Parent CVs:", len(parents))
    print("Full mention matches:", len(verified))
    print("Candidates sent to LLM:", len(selected))

    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
    )

    results = []

    for candidate_id in selected:
        print(f"\n[ASSESS] {candidate_id}")

        assessments = assess_candidate(
            llm,
            candidate_id,
            parents[candidate_id],
            required_skills,
        )

        scores = [
            item["score"]
            for item in assessments
        ]

        # Conservative ranking: the weakest required
        # skill is the primary evidence-strength signal.
        min_score = min(scores) if scores else 0
        total_score = sum(scores)

        result = {
            "candidate_id": candidate_id,
            "bm25_score": bm25_scores[candidate_id],
            "minimum_skill_score": min_score,
            "total_evidence_score": total_score,
            "assessments": assessments,
        }

        results.append(result)

        for item in assessments:
            print(
                f"  {item['skill']}: "
                f"{item['validated_level']} "
                f"(score={item['score']})"
            )

            for warning in item["warnings"]:
                print("    WARNING:", warning)

    # Evidence strength first, BM25 as tie-breaker.
    results.sort(
        key=lambda item: (
            -item["minimum_skill_score"],
            -item["total_evidence_score"],
            -item["bm25_score"],
            item["candidate_id"],
        )
    )

    print("\n=== FINAL EVIDENCE-AWARE RANKING ===")

    for position, result in enumerate(
        results[:TOP_K],
        start=1,
    ):
        print(
            f"{position}. {result['candidate_id']} "
            f"| min={result['minimum_skill_score']} "
            f"| total={result['total_evidence_score']} "
            f"| BM25={result['bm25_score']:.3f}"
        )

    print("\n=== ORIGINAL BM25 TOP 5 ===")
    print(bm25_ranked[:TOP_K])

    print("\n=== EVIDENCE-AWARE TOP 5 ===")
    print([
        item["candidate_id"]
        for item in results[:TOP_K]
    ])

    output = {
        "lesson": "4H",
        "query": query,
        "required_skills": required_skills,
        "bm25_top_5": bm25_ranked[:TOP_K],
        "evidence_aware_top_5": [
            item["candidate_id"]
            for item in results[:TOP_K]
        ],
        "ranked_candidates": results,
        "limitations": [
            "LLM evidence classification is not ground truth.",
            "Exact quote validation cannot verify the "
            "meaning of a context quotation.",
            "Only the top verified candidates are graded.",
            "Evidence scores are experimental.",
            "This is not a validated hiring suitability score.",
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

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
