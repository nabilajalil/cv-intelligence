
"""Lesson 5F.3: Deterministic evidence rendering for GraphRAG.

Key idea:
    Python selects and preserves source evidence.
    The LLM writes optional explanations using evidence IDs.
    Python validates references and renders exact citations.

Pipeline:
    retrieve -> verify -> collect -> explain -> validate -> render

No graph modifications.
One optional LLM call when candidates are found.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
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

from scripts.evaluate_graph_hybrid_retrieval import (
    build_bm25,
    rank_candidates_bm25,
    extract_skills,
    graph_verified_candidates,
    filter_ranking,
    load_graph,
)

from scripts.graphrag.demo_validated_graph_answers import (
    normalize_retrieval_query,
    extract_skill_quote,
)


load_dotenv()

TOP_K = 5

OUTPUT_DIR = Path(
    "data/public_benchmark/graph/"
    "deterministic_answers"
)


class Explanation(BaseModel):
    candidate_id: str
    statement: str
    evidence_ids: list[str] = Field(
        default_factory=list
    )


class ExplanationBatch(BaseModel):
    explanations: list[Explanation] = Field(
        default_factory=list
    )


class GraphState(TypedDict, total=False):
    query: str
    retrieval_query: str
    required_skills: list[str]
    ranking: list[str]
    selected: list[str]
    evidence_registry: dict
    candidate_evidence_ids: dict
    raw_explanations: list[dict]
    accepted_explanations: list[dict]
    rejected_explanations: list[dict]
    rendered_answer: str
    skill_coverage: dict


def stable_query_id(query):
    return hashlib.sha256(
        query.encode("utf-8")
    ).hexdigest()[:12]


def make_evidence_id(
    candidate_id,
    evidence_type,
    chunk_id,
    subject,
):
    value = (
        f"{candidate_id}|{evidence_type}|"
        f"{chunk_id}|{subject}"
    )

    return (
        "ev_"
        + hashlib.sha256(
            value.encode("utf-8")
        ).hexdigest()[:12]
    )


def valid_source_quote(
    candidate_id,
    chunk_id,
    quote,
    chunk_lookup,
):
    chunk = chunk_lookup.get(chunk_id)

    return (
        chunk is not None
        and chunk["candidate_id"] == candidate_id
        and isinstance(quote, str)
        and bool(quote.strip())
        and quote in chunk["text"]
    )


def add_evidence(
    registry,
    candidate_map,
    chunk_lookup,
    candidate_id,
    evidence_type,
    subject,
    chunk_id,
    quote,
    metadata=None,
):
    """Insert only evidence with an exact source quotation."""

    if not valid_source_quote(
        candidate_id,
        chunk_id,
        quote,
        chunk_lookup,
    ):
        return None

    evidence_id = make_evidence_id(
        candidate_id,
        evidence_type,
        chunk_id,
        subject,
    )

    record = {
        "evidence_id": evidence_id,
        "candidate_id": candidate_id,
        "type": evidence_type,
        "subject": subject,
        "chunk_id": chunk_id,
        "quote": quote,
        "metadata": metadata or {},
    }

    registry[evidence_id] = record

    candidate_map.setdefault(
        candidate_id,
        [],
    ).append(evidence_id)

    return evidence_id


def collect_evidence_registry(
    selected,
    required_skills,
    parents,
    graph,
    chunk_lookup,
):
    registry = {}
    candidate_map = {}

    for candidate_id in selected:
        candidate_map[candidate_id] = []

        skill_matches = find_skill_evidence(
            parents[candidate_id],
            required_skills,
        )

        for skill in required_skills:
            matching_chunks = skill_matches.get(
                skill,
                [],
            )

            if not matching_chunks:
                continue

            chunk_id = matching_chunks[0]
            chunk = chunk_lookup[chunk_id]

            quote = extract_skill_quote(
                chunk["text"],
                skill,
            )

            add_evidence(
                registry,
                candidate_map,
                chunk_lookup,
                candidate_id,
                "skill",
                skill,
                chunk_id,
                quote,
            )

        candidate_node = (
            f"candidate:{candidate_id}"
        )

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

                role = node.get("role", "")
                employer = node.get(
                    "employer",
                    "",
                )

                add_evidence(
                    registry,
                    candidate_map,
                    chunk_lookup,
                    candidate_id,
                    "employment",
                    f"{role} at {employer}",
                    edge.get("chunk_id"),
                    edge.get("quote"),
                    metadata={
                        "role": role,
                        "employer": employer,
                    },
                )

            elif relation == "CONTRIBUTED_TO":
                node = graph.nodes[neighbor]

                project = node.get(
                    "label",
                    "",
                )

                add_evidence(
                    registry,
                    candidate_map,
                    chunk_lookup,
                    candidate_id,
                    "project",
                    project,
                    edge.get("chunk_id"),
                    edge.get("quote"),
                )

    return registry, candidate_map


def validate_explanations(
    explanations,
    selected,
    registry,
):
    accepted = []
    rejected = []

    seen = set()

    for explanation in explanations:
        reasons = []

        if explanation.candidate_id not in selected:
            reasons.append("UNKNOWN_CANDIDATE")

        if not explanation.statement.strip():
            reasons.append("EMPTY_STATEMENT")

        if not explanation.evidence_ids:
            reasons.append("NO_EVIDENCE_IDS")

        for evidence_id in explanation.evidence_ids:
            record = registry.get(evidence_id)

            if record is None:
                reasons.append(
                    f"UNKNOWN_EVIDENCE_ID:{evidence_id}"
                )

            elif (
                record["candidate_id"]
                != explanation.candidate_id
            ):
                reasons.append(
                    f"WRONG_CANDIDATE:{evidence_id}"
                )

        key = (
            explanation.candidate_id,
            explanation.statement,
            tuple(explanation.evidence_ids),
        )

        if key in seen:
            reasons.append("DUPLICATE_EXPLANATION")

        seen.add(key)

        item = explanation.model_dump()

        if reasons:
            item["reasons"] = reasons
            rejected.append(item)
        else:
            accepted.append(item)

    return accepted, rejected


def compute_skill_coverage(
    selected,
    required_skills,
    registry,
):
    expected = len(selected) * len(
        required_skills
    )

    covered = 0
    missing = []

    for candidate_id in selected:
        for skill in required_skills:
            found = any(
                record["candidate_id"]
                == candidate_id
                and record["type"] == "skill"
                and record["subject"] == skill
                for record in registry.values()
            )

            if found:
                covered += 1
            else:
                missing.append(
                    {
                        "candidate_id": candidate_id,
                        "skill": skill,
                    }
                )

    return {
        "expected": expected,
        "covered": covered,
        "missing": missing,
        "complete": covered == expected,
    }


def render_answer(
    selected,
    required_skills,
    registry,
    candidate_map,
    accepted_explanations,
    coverage,
):
    lines = []

    if not selected:
        return (
            "No candidates were verified as "
            "mentioning all required skills."
        )

    lines.append(
        "Candidates verified for: "
        + ", ".join(required_skills)
    )

    for index, candidate_id in enumerate(
        selected,
        start=1,
    ):
        lines.append(
            f"\n{index}. Candidate {candidate_id}"
        )

        evidence_ids = candidate_map.get(
            candidate_id,
            [],
        )

        candidate_records = [
            registry[evidence_id]
            for evidence_id in evidence_ids
        ]

        skill_records = [
            record
            for record in candidate_records
            if record["type"] == "skill"
        ]

        employment_records = [
            record
            for record in candidate_records
            if record["type"] == "employment"
        ]

        project_records = [
            record
            for record in candidate_records
            if record["type"] == "project"
        ]

        lines.append(
            "  Mandatory skill evidence:"
        )

        for skill in required_skills:
            matching = [
                record
                for record in skill_records
                if record["subject"] == skill
            ]

            if not matching:
                lines.append(
                    f"    {skill}: NO VALIDATED CITATION"
                )
                continue

            record = matching[0]

            lines.append(
                f"    {skill}: "
                f"[{record['chunk_id']}]"
            )

            lines.append(
                f"      Exact quote: "
                f"{record['quote']!r}"
            )

        if employment_records:
            lines.append(
                "  Structured employment:"
            )

            for record in employment_records:
                lines.append(
                    f"    {record['subject']} "
                    f"[{record['chunk_id']}]"
                )
        else:
            lines.append(
                "  Structured employment: "
                "not yet extracted."
            )

        if project_records:
            lines.append(
                "  Accepted projects:"
            )

            for record in project_records:
                lines.append(
                    f"    {record['subject']} "
                    f"[{record['chunk_id']}]"
                )
        else:
            lines.append(
                "  Accepted projects: "
                "not yet extracted."
            )

        candidate_explanations = [
            item
            for item in accepted_explanations
            if item["candidate_id"]
            == candidate_id
        ]

        if candidate_explanations:
            lines.append(
                "  LLM explanations "
                "(reference-validated, "
                "not semantically verified):"
            )

            for item in candidate_explanations:
                citations = []

                for evidence_id in item[
                    "evidence_ids"
                ]:
                    record = registry[evidence_id]

                    citations.append(
                        record["chunk_id"]
                    )

                lines.append(
                    f"    {item['statement']}"
                )

                lines.append(
                    "    Cited chunks: "
                    + ", ".join(
                        dict.fromkeys(citations)
                    )
                )

    lines.append(
        "\nSkill evidence coverage: "
        f"{coverage['covered']}/"
        f"{coverage['expected']}"
    )

    return "\n".join(lines)


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
        ExplanationBatch
    )

    def retrieve(state: GraphState):
        query = state["query"]

        retrieval_query = (
            normalize_retrieval_query(query)
        )

        required_skills = extract_skills(
            query
        )

        ranking = rank_candidates_bm25(
            retrieval_query,
            chunks,
            bm25,
        )

        print("\n[RETRIEVE]")
        print(
            "Retrieval query:",
            retrieval_query,
        )
        print(
            "Required skills:",
            required_skills,
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
            "Graph-verified:",
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

        registry, candidate_map = (
            collect_evidence_registry(
                state["selected"],
                state["required_skills"],
                parents,
                graph,
                chunk_lookup,
            )
        )

        coverage = compute_skill_coverage(
            state["selected"],
            state["required_skills"],
            registry,
        )

        print(
            "Evidence records:",
            len(registry),
        )

        print(
            "Mandatory skill coverage:",
            f"{coverage['covered']}/"
            f"{coverage['expected']}",
        )

        return {
            "evidence_registry": registry,
            "candidate_evidence_ids": (
                candidate_map
            ),
            "skill_coverage": coverage,
        }

    def explain(state: GraphState):
        print("\n[EXPLAIN]")

        if not state["selected"]:
            return {
                "raw_explanations": []
            }

        prompt = f"""
You are a cautious recruiting research assistant.

RECRUITER QUESTION:
{state["query"]}

CANDIDATES IN ORDER:
{state["selected"]}

EVIDENCE REGISTRY:
{json.dumps(state["evidence_registry"], indent=2)}

Create at most two short explanations per candidate.

RULES:
- Only reference evidence IDs from the registry.
- Do not copy quotations.
- Do not invent candidate details.
- A skill mention does not establish proficiency.
- Do not connect skills to projects or jobs
  unless the evidence explicitly establishes it.
- Each explanation must cite its evidence IDs.
- Prefer factual, cautious statements.
- Return no explanations if evidence is insufficient.
"""

        response = llm.invoke(prompt)

        return {
            "raw_explanations": [
                item.model_dump()
                for item in response.explanations
            ]
        }

    def validate(state: GraphState):
        print("\n[VALIDATE REFERENCES]")

        explanations = [
            Explanation.model_validate(item)
            for item in state[
                "raw_explanations"
            ]
        ]

        accepted, rejected = (
            validate_explanations(
                explanations,
                state["selected"],
                state["evidence_registry"],
            )
        )

        print(
            "Accepted explanations:",
            len(accepted),
        )

        print(
            "Rejected explanations:",
            len(rejected),
        )

        return {
            "accepted_explanations": accepted,
            "rejected_explanations": rejected,
        }

    def render(state: GraphState):
        print("\n[RENDER]")

        answer = render_answer(
            state["selected"],
            state["required_skills"],
            state["evidence_registry"],
            state["candidate_evidence_ids"],
            state["accepted_explanations"],
            state["skill_coverage"],
        )

        return {
            "rendered_answer": answer
        }

    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("verify", verify)
    workflow.add_node("collect", collect)
    workflow.add_node("explain", explain)
    workflow.add_node("validate", validate)
    workflow.add_node("render", render)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "verify")
    workflow.add_edge("verify", "collect")
    workflow.add_edge("collect", "explain")
    workflow.add_edge("explain", "validate")
    workflow.add_edge("validate", "render")
    workflow.add_edge("render", END)

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
        "=== LESSON 5F.3: "
        "DETERMINISTIC GRAPH ANSWERS ==="
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
        {"query": args.query}
    )

    print("\n" + "=" * 65)
    print("FINAL ANSWER")
    print("=" * 65)
    print(result["rendered_answer"])

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    coverage = result["skill_coverage"]

    print(
        "Mandatory skill coverage:",
        f"{coverage['covered']}/"
        f"{coverage['expected']}",
    )

    print(
        "Coverage complete:",
        coverage["complete"],
    )

    print(
        "Accepted explanations:",
        len(result["accepted_explanations"]),
    )

    print(
        "Rejected explanations:",
        len(result["rejected_explanations"]),
    )

    if result["rejected_explanations"]:
        print("\nRejected explanation details:")

        for item in result[
            "rejected_explanations"
        ]:
            print(
                item["candidate_id"],
                item["reasons"],
            )

    output = {
        "query": args.query,
        "retrieval_query": result[
            "retrieval_query"
        ],
        "required_skills": result[
            "required_skills"
        ],
        "selected_candidates": result[
            "selected"
        ],
        "skill_coverage": coverage,
        "evidence_registry": result[
            "evidence_registry"
        ],
        "accepted_explanations": result[
            "accepted_explanations"
        ],
        "rejected_explanations": result[
            "rejected_explanations"
        ],
        "rendered_answer": result[
            "rendered_answer"
        ],
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / (
        stable_query_id(args.query)
        + ".json"
    )

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output_path)

    print(
        "\n=== LESSON 5F.3 COMPLETE ==="
    )


if __name__ == "__main__":
    main()
