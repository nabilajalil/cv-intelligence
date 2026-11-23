
"""Lesson 5F.2: Validated GraphRAG answers.

Features:
- Normalize simple mandatory-skill queries
- Rank candidates with BM25
- Verify skills through NetworkX
- Collect original source evidence
- Generate structured LLM claims
- Validate candidate IDs, chunk IDs and exact quotations
- Reject unsupported or incorrectly cited claims

No graph modifications.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from scripts.demo_parent_cv_verification import (
    load_chunks,
    collect_candidate_chunks,
    find_skill_evidence,
)

from scripts.graphrag.evaluate_graph_hybrid_retrieval import (
    build_bm25,
    rank_candidates_bm25,
    extract_skills,
    graph_verified_candidates,
    filter_ranking,
    load_graph,
)


load_dotenv()

TOP_K = 5

OUTPUT_PATH = Path(
    "data/public_benchmark/graph/"
    "validated_graph_answers.json"
)


class EvidenceClaim(BaseModel):
    candidate_id: str
    claim_type: Literal[
        "skill",
        "employment",
        "project",
    ]
    statement: str
    chunk_id: str
    quote: str


class CandidateAnswer(BaseModel):
    candidate_id: str
    claims: list[EvidenceClaim] = Field(
        default_factory=list
    )


class StructuredAnswer(BaseModel):
    candidates: list[CandidateAnswer] = Field(
        default_factory=list
    )


class GraphState(TypedDict, total=False):
    query: str
    retrieval_query: str
    required_skills: list[str]
    ranking: list[str]
    selected: list[str]
    evidence: list[dict]
    raw_answer: dict
    validated_answer: list[dict]
    rejected_claims: list[dict]


def normalize_retrieval_query(query):
    """Use canonical retrieval text for known skill queries.

    This is a controlled benchmark normalization,
    not a general natural-language query parser.
    """

    skills = extract_skills(query)

    if not skills:
        return query

    if len(skills) == 1:
        return (
            "Find candidates with "
            + skills[0]
        )

    return (
        "Find candidates with "
        + ", ".join(skills[:-1])
        + " and "
        + skills[-1]
    )


def extract_skill_quote(text, skill, window=110):
    """Return a short source excerpt containing the skill."""

    pattern = (
        r"(?<![A-Za-z0-9])"
        + re.escape(skill)
        + r"(?![A-Za-z0-9])"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return ""

    start = max(
        0,
        match.start() - window,
    )

    end = min(
        len(text),
        match.end() + window,
    )

    return text[start:end]


def collect_candidate_evidence(
    candidate_id,
    required_skills,
    parents,
    graph,
    chunk_lookup,
):
    """Collect verified source material for one candidate."""

    candidate_node = f"candidate:{candidate_id}"

    skill_matches = find_skill_evidence(
        parents[candidate_id],
        required_skills,
    )

    skills = []

    for skill in required_skills:
        chunk_ids = skill_matches.get(
            skill,
            [],
        )

        if not chunk_ids:
            continue

        chunk_id = chunk_ids[0]
        chunk = chunk_lookup[chunk_id]

        quote = extract_skill_quote(
            chunk["text"],
            skill,
        )

        if quote:
            skills.append(
                {
                    "skill": skill,
                    "chunk_id": chunk_id,
                    "quote": quote,
                }
            )

    employment = []
    projects = []

    for neighbor in graph.successors(
        candidate_node
    ):
        edge = graph.edges[
            candidate_node,
            neighbor,
        ]

        relation = edge.get("relation")

        if relation == "HELD_POSITION":
            node = graph.nodes[neighbor]

            employment.append(
                {
                    "role": node.get("role"),
                    "employer": node.get("employer"),
                    "chunk_id": edge.get("chunk_id"),
                    "quote": edge.get("quote"),
                }
            )

        elif relation == "CONTRIBUTED_TO":
            node = graph.nodes[neighbor]

            projects.append(
                {
                    "project": node.get("label"),
                    "chunk_id": edge.get("chunk_id"),
                    "quote": edge.get("quote"),
                }
            )

    return {
        "candidate_id": candidate_id,
        "skills": skills,
        "employment": employment,
        "projects": projects,
        "structured_history_available": bool(
            employment or projects
        ),
    }


def validate_claim(
    claim,
    selected,
    chunk_lookup,
    evidence_by_candidate,
):
    """Validate source provenance and allowed evidence.

    These checks establish citation integrity.
    They do not prove that every paraphrase is
    semantically entailed by the quotation.
    """

    reasons = []

    candidate_id = claim.candidate_id
    chunk_id = claim.chunk_id
    quote = claim.quote

    if candidate_id not in selected:
        reasons.append("UNKNOWN_CANDIDATE")

    chunk = chunk_lookup.get(chunk_id)

    if chunk is None:
        reasons.append("UNKNOWN_CHUNK")
    else:
        if chunk["candidate_id"] != candidate_id:
            reasons.append("WRONG_CANDIDATE_CHUNK")

        if not quote or quote not in chunk["text"]:
            reasons.append("QUOTE_NOT_IN_SOURCE")

    candidate_evidence = evidence_by_candidate.get(
        candidate_id
    )

    if candidate_evidence is None:
        reasons.append("NO_CANDIDATE_EVIDENCE")
    else:
        allowed_records = candidate_evidence.get(
            {
                "skill": "skills",
                "employment": "employment",
                "project": "projects",
            }[claim.claim_type],
            [],
        )

        allowed = any(
            record["chunk_id"] == chunk_id
            and quote
            and quote in record["quote"]
            for record in allowed_records
        )

        if not allowed:
            reasons.append(
                "QUOTE_NOT_IN_ALLOWED_EVIDENCE"
            )

    if not claim.statement.strip():
        reasons.append("EMPTY_STATEMENT")

    return reasons


def build_app(chunks, parents, graph):
    bm25 = build_bm25(chunks)

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    ).with_structured_output(
        StructuredAnswer
    )

    def retrieve(state: GraphState):
        query = state["query"]

        required_skills = extract_skills(
            query
        )

        retrieval_query = (
            normalize_retrieval_query(query)
        )

        ranking = rank_candidates_bm25(
            retrieval_query,
            chunks,
            bm25,
        )

        print("\n[RETRIEVE]")
        print("Original query:", query)
        print(
            "Retrieval query:",
            retrieval_query,
        )
        print(
            "Required skills:",
            required_skills,
        )
        print(
            "BM25 top 5:",
            ranking[:TOP_K],
        )

        return {
            "retrieval_query": retrieval_query,
            "required_skills": required_skills,
            "ranking": ranking,
        }

    def verify(state: GraphState):
        allowed = graph_verified_candidates(
            graph,
            state["required_skills"],
        )

        selected = filter_ranking(
            state["ranking"],
            allowed,
        )[:TOP_K]

        print("\n[VERIFY]")
        print(
            "Graph-verified candidates:",
            len(allowed),
        )
        print(
            "Selected:",
            selected,
        )

        return {
            "selected": selected,
        }

    def collect(state: GraphState):
        print("\n[COLLECT EVIDENCE]")

        records = [
            collect_candidate_evidence(
                candidate_id,
                state["required_skills"],
                parents,
                graph,
                chunk_lookup,
            )
            for candidate_id in state["selected"]
        ]

        for record in records:
            print(
                record["candidate_id"],
                "| skills:",
                len(record["skills"]),
                "| employment:",
                len(record["employment"]),
                "| projects:",
                len(record["projects"]),
            )

        return {
            "evidence": records,
        }

    def generate(state: GraphState):
        print("\n[GENERATE]")

        if not state["selected"]:
            return {
                "raw_answer": {
                    "candidates": []
                }
            }

        prompt = f"""
You are a cautious recruiting research assistant.

RECRUITER QUESTION:
{state["query"]}

SELECTED CANDIDATES IN RANKING ORDER:
{state["selected"]}

SOURCE EVIDENCE:
{json.dumps(state["evidence"], indent=2)}

Produce structured candidate claims.

RULES:
- Only use the supplied evidence.
- Preserve candidate ranking order.
- Include one claim per required skill.
- You may include employment and project claims
  only when their records are supplied.
- Every claim must include an exact quotation
  copied from its supporting evidence record.
- Every claim must use the matching chunk ID.
- The statement must not be stronger than the quote.
- A technology mention is not proof of proficiency,
  professional use or years of experience.
- Do not link a skill to a job or project unless
  the source explicitly establishes that link.
- Do not invent names, projects or responsibilities.
- Prefer short, factual statements.
- Do not claim that missing structured records
  mean a candidate has no experience.
"""

        response = llm.invoke(prompt)

        return {
            "raw_answer": response.model_dump()
        }

    def validate(state: GraphState):
        print("\n[VALIDATE CLAIMS]")

        evidence_by_candidate = {
            record["candidate_id"]: record
            for record in state["evidence"]
        }

        raw = StructuredAnswer.model_validate(
            state["raw_answer"]
        )

        accepted = []
        rejected = []
        seen_candidates = set()

        for candidate_answer in raw.candidates:
            candidate_id = (
                candidate_answer.candidate_id
            )

            if candidate_id in seen_candidates:
                print(
                    "Duplicate candidate entry:",
                    candidate_id,
                )

            seen_candidates.add(candidate_id)

            for claim in candidate_answer.claims:
                reasons = validate_claim(
                    claim,
                    state["selected"],
                    chunk_lookup,
                    evidence_by_candidate,
                )

                if (
                    claim.candidate_id
                    != candidate_id
                ):
                    reasons.append(
                        "CANDIDATE_ID_MISMATCH"
                    )

                record = claim.model_dump()

                if reasons:
                    record["reasons"] = reasons
                    rejected.append(record)
                else:
                    accepted.append(record)

        print(
            "Accepted claims:",
            len(accepted),
        )
        print(
            "Rejected claims:",
            len(rejected),
        )

        return {
            "validated_answer": accepted,
            "rejected_claims": rejected,
        }

    workflow = StateGraph(GraphState)

    workflow.add_node(
        "retrieve",
        retrieve,
    )
    workflow.add_node(
        "verify",
        verify,
    )
    workflow.add_node(
        "collect",
        collect,
    )
    workflow.add_node(
        "generate",
        generate,
    )
    workflow.add_node(
        "validate",
        validate,
    )

    workflow.add_edge(
        START,
        "retrieve",
    )
    workflow.add_edge(
        "retrieve",
        "verify",
    )
    workflow.add_edge(
        "verify",
        "collect",
    )
    workflow.add_edge(
        "collect",
        "generate",
    )
    workflow.add_edge(
        "generate",
        "validate",
    )
    workflow.add_edge(
        "validate",
        END,
    )

    return workflow.compile()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "query",
        nargs="?",
        default=(
            "Find candidates with Python and SQL, "
            "and explain their relevant experience"
        ),
    )

    args = parser.parse_args()

    print(
        "=== LESSON 5F.2: VALIDATED GRAPH ANSWERS ==="
    )

    chunks = load_chunks()
    parents = collect_candidate_chunks(chunks)
    graph = load_graph()

    print("CV chunks:", len(chunks))
    print("Candidates:", len(parents))
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    app = build_app(
        chunks,
        parents,
        graph,
    )

    result = app.invoke(
        {
            "query": args.query
        }
    )

    print("\n" + "=" * 65)
    print("VALIDATED ANSWER")
    print("=" * 65)

    selected = result["selected"]
    accepted = result["validated_answer"]
    rejected = result["rejected_claims"]

    if not selected:
        print(
            "No candidates were verified as "
            "mentioning all required skills."
        )
    else:
        for candidate_id in selected:
            print(
                f"\nCandidate {candidate_id}"
            )

            candidate_claims = [
                claim
                for claim in accepted
                if claim["candidate_id"]
                == candidate_id
            ]

            if not candidate_claims:
                print(
                    "  No validated claims returned."
                )

            for claim in candidate_claims:
                print(
                    f"  [{claim['claim_type']}] "
                    f"{claim['statement']}"
                )
                print(
                    f"    Citation: "
                    f"{claim['chunk_id']}"
                )
                print(
                    f"    Quote: "
                    f"{claim['quote']}"
                )

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(
        "Selected candidates:",
        len(selected),
    )
    print(
        "Accepted claims:",
        len(accepted),
    )
    print(
        "Rejected claims:",
        len(rejected),
    )

    if rejected:
        print("\nRejected claim details:")

        for claim in rejected:
            print(
                f"  {claim['candidate_id']} "
                f"[{claim['claim_type']}] "
                f"{claim['reasons']}"
            )

    output = {
        "query": args.query,
        "retrieval_query": result[
            "retrieval_query"
        ],
        "required_skills": result[
            "required_skills"
        ],
        "selected_candidates": selected,
        "accepted_claims": accepted,
        "rejected_claims": rejected,
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
    print(
        "\n=== LESSON 5F.2 COMPLETE ==="
    )


if __name__ == "__main__":
    main()

