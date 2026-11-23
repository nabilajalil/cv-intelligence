
"""Lesson 5C: LLM-based CV entity and relationship extraction.

Extract:
- Job roles
- Employers
- Projects
- Technologies explicitly linked to projects

Validate each extracted relationship using an exact quote
from the original source chunk.

Important:
An exact quote confirms that the text exists.
It does not independently prove the LLM interpreted
the relationship correctly.

This lesson does NOT modify the existing knowledge graph.
"""

import json
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from scripts.retrieval.demo_parent_cv_verification import (
    load_chunks,
    collect_candidate_chunks,
)


load_dotenv()


OUTPUT_DIR = Path(
    "data/public_benchmark/graph/entity_extraction"
)

CANDIDATE_IDS = [
    "r00041",
    "r00049",
    "r00001",
]

MAX_CHUNKS_PER_CANDIDATE = 10


class ExtractedRelationship(BaseModel):
    """One relationship extracted from CV text."""

    relation: Literal[
        "HELD_ROLE",
        "WORKED_AT",
        "CONTRIBUTED_TO",
        "PROJECT_USED_TECHNOLOGY",
    ]

    source: str = Field(
        description="Source entity name."
    )

    target: str = Field(
        description="Target entity name."
    )

    chunk_id: str = Field(
        description="Exact source chunk ID."
    )

    quote: str = Field(
        description="Exact verbatim quote supporting the relationship."
    )

    explanation: str = Field(
        description="Brief explanation of the relationship."
    )


class CandidateExtraction(BaseModel):
    """Structured extraction for one candidate."""

    relationships: list[ExtractedRelationship] = Field(
        default_factory=list
    )


SYSTEM_PROMPT = """
You extract evidence-backed knowledge graph relationships
from a candidate CV.

The candidate ID is: {candidate_id}

Allowed relationships:

1. HELD_ROLE
   Source: candidate ID
   Target: explicitly stated job title

2. WORKED_AT
   Source: candidate ID
   Target: explicitly stated employer

3. CONTRIBUTED_TO
   Source: candidate ID
   Target: explicitly named or clearly described project

4. PROJECT_USED_TECHNOLOGY
   Source: project name
   Target: technology explicitly associated with that project

Rules:
- Extract only relationships explicitly supported by the CV.
- Do not invent employers, projects, roles or technologies.
- Do not treat a general skills list as project-specific use.
- Do not assume a candidate used a technology professionally
  just because it appears in their skills list.
- A technology must be linked to the particular project
  in the evidence, not merely appear elsewhere in the CV.
- For each relationship, cite exactly one supplied chunk ID.
- Provide a short, exact, verbatim supporting quote.
- The quote must come from the cited chunk.
- For candidate relationships, source must be the candidate ID.
- For project-technology relationships, source must be
  the project name, not the candidate ID.
- If evidence is ambiguous, omit the relationship.
- Return an empty list when nothing is supported.
"""


def format_candidate_chunks(candidate_chunks):
    """Format chunks with explicit citation markers."""

    parts = []

    for chunk in candidate_chunks:
        parts.append(
            f"CHUNK_ID: {chunk['chunk_id']}\n"
            f"TEXT:\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(parts)


def extract_candidate(llm, candidate_id, candidate_chunks):
    """Ask the LLM to extract structured relationships."""

    source_text = format_candidate_chunks(
        candidate_chunks
    )

    messages = [
        (
            "system",
            SYSTEM_PROMPT.format(
                candidate_id=candidate_id
            ),
        ),
        (
            "human",
            "Extract knowledge graph relationships "
            "from the following CV chunks:\n\n"
            + source_text,
        ),
    ]

    return llm.invoke(messages)


def validate_relationship(
    relationship,
    candidate_id,
    chunk_lookup,
):
    """Validate citation and basic relationship structure."""

    warnings = []

    chunk = chunk_lookup.get(
        relationship.chunk_id
    )

    if chunk is None:
        warnings.append(
            "Referenced chunk does not exist."
        )

        return False, warnings

    if chunk["candidate_id"] != candidate_id:
        warnings.append(
            "Referenced chunk belongs to another candidate."
        )

    if not relationship.quote.strip():
        warnings.append(
            "Supporting quote is empty."
        )

    elif relationship.quote not in chunk["text"]:
        warnings.append(
            "Supporting quote is not an exact substring."
        )

    candidate_relations = {
        "HELD_ROLE",
        "WORKED_AT",
        "CONTRIBUTED_TO",
    }

    if relationship.relation in candidate_relations:
        if relationship.source != candidate_id:
            warnings.append(
                "Candidate relationship has incorrect source."
            )

    if relationship.relation == "PROJECT_USED_TECHNOLOGY":
        if not relationship.source.strip():
            warnings.append(
                "Project source is empty."
            )

        if not relationship.target.strip():
            warnings.append(
                "Technology target is empty."
            )

    return len(warnings) == 0, warnings


def main():
    print(
        "=== LESSON 5C: LLM GRAPH ENTITY EXTRACTION ==="
    )

    chunks = load_chunks()
    parents = collect_candidate_chunks(chunks)

    print("Loaded chunks:", len(chunks))
    print("Available candidates:", len(parents))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    ).with_structured_output(
        CandidateExtraction
    )

    all_results = []

    for candidate_id in CANDIDATE_IDS:
        print("\n" + "=" * 65)
        print("Candidate:", candidate_id)
        print("=" * 65)

        candidate_chunks = parents.get(
            candidate_id,
            [],
        )[:MAX_CHUNKS_PER_CANDIDATE]

        if not candidate_chunks:
            print("No chunks found.")
            continue

        print(
            "Chunks supplied to LLM:",
            len(candidate_chunks),
        )

        extraction = extract_candidate(
            llm,
            candidate_id,
            candidate_chunks,
        )

        chunk_lookup = {
            chunk["chunk_id"]: chunk
            for chunk in candidate_chunks
        }

        accepted = []
        rejected = []

        for relationship in extraction.relationships:
            valid, warnings = validate_relationship(
                relationship,
                candidate_id,
                chunk_lookup,
            )

            record = relationship.model_dump()
            record["valid"] = valid
            record["warnings"] = warnings

            if valid:
                accepted.append(record)
                status = "VALID"
            else:
                rejected.append(record)
                status = "REJECTED"

            print(
                f"\n[{status}] "
                f"{relationship.source} "
                f"--[{relationship.relation}]--> "
                f"{relationship.target}"
            )

            print(
                "  Chunk:",
                relationship.chunk_id,
            )

            print(
                "  Quote:",
                relationship.quote,
            )

            for warning in warnings:
                print("  WARNING:", warning)

        print(
            "\nAccepted relationships:",
            len(accepted),
        )

        print(
            "Rejected relationships:",
            len(rejected),
        )

        all_results.append(
            {
                "candidate_id": candidate_id,
                "accepted": accepted,
                "rejected": rejected,
            }
        )

    output_path = OUTPUT_DIR / "results.json"

    output_path.write_text(
        json.dumps(
            all_results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)

    print("Saved results:", output_path)


if __name__ == "__main__":
    main()

