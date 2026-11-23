
"""Lesson 5F: Graph-grounded candidate answers.

Pipeline:
    retrieve -> verify -> collect_evidence -> generate

Uses:
- BM25 ranking from Lesson 5E
- Enriched NetworkX graph from Lesson 5D
- Original CV chunks for citations
- LangGraph for orchestration
- gpt-4.1-mini for answer generation

The graph is read-only.
Only the generate node calls OpenAI.
"""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import networkx as nx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

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
)


GRAPH_PATH = Path(
    "data/public_benchmark/graph/enriched/"
    "enriched_candidate_graph.json"
)

TOP_K = 5
MAX_EVIDENCE_CHARS = 450


class GraphRAGState(TypedDict, total=False):
    query: str
    required_skills: list[str]
    bm25_ranking: list[str]
    verified_candidates: list[str]
    evidence: list[dict]
    answer: str


def load_graph():
    data = json.loads(
        GRAPH_PATH.read_text(encoding="utf-8")
    )

    return nx.node_link_graph(
        data,
        directed=True,
        multigraph=False,
        edges="links",
    )


def make_app(chunks, parents, graph):
    bm25 = build_bm25(chunks)

    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0,
    )

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    def retrieve(state: GraphRAGState):
        query = state["query"]

        required_skills = extract_skills(query)

        ranking = rank_candidates_bm25(
            query,
            chunks,
            bm25,
        )

        print("\n[RETRIEVE]")
        print("Query:", query)
        print("Required skills:", required_skills)
        print("BM25 top 5:", ranking[:TOP_K])

        return {
            "required_skills": required_skills,
            "bm25_ranking": ranking,
        }

    def verify(state: GraphRAGState):
        allowed = graph_verified_candidates(
            graph,
            state["required_skills"],
        )

        verified_ranking = filter_ranking(
            state["bm25_ranking"],
            allowed,
        )

        selected = verified_ranking[:TOP_K]

        print("\n[VERIFY]")
        print("Graph-verified total:", len(allowed))
        print("Selected candidates:", selected)

        return {
            "verified_candidates": selected,
        }

    def collect_evidence(state: GraphRAGState):
        print("\n[COLLECT EVIDENCE]")

        all_evidence = []

        for candidate_id in state[
            "verified_candidates"
        ]:
            candidate_node = (
                f"candidate:{candidate_id}"
            )

            candidate_chunks = parents[candidate_id]

            skill_evidence = find_skill_evidence(
                candidate_chunks,
                state["required_skills"],
            )

            skill_records = []

            for skill in state["required_skills"]:
                matching_ids = skill_evidence.get(
                    skill,
                    [],
                )

                if not matching_ids:
                    continue

                # Select one verified source chunk.
                chunk_id = matching_ids[0]
                chunk = chunk_lookup[chunk_id]

                skill_records.append(
                    {
                        "skill": skill,
                        "chunk_id": chunk_id,
                        "text": chunk["text"][
                            :MAX_EVIDENCE_CHARS
                        ],
                    }
                )

            employment_records = []
            project_records = []

            for neighbor in graph.successors(
                candidate_node
            ):
                edge = graph.edges[
                    candidate_node,
                    neighbor,
                ]

                relation = edge.get("relation")

                if relation == "HELD_POSITION":
                    employment_node = graph.nodes[
                        neighbor
                    ]

                    employment_records.append(
                        {
                            "role": employment_node.get(
                                "role"
                            ),
                            "employer": employment_node.get(
                                "employer"
                            ),
                            "chunk_id": edge.get(
                                "chunk_id"
                            ),
                            "quote": edge.get(
                                "quote"
                            ),
                        }
                    )

                elif relation == "CONTRIBUTED_TO":
                    project_node = graph.nodes[
                        neighbor
                    ]

                    project_records.append(
                        {
                            "project": project_node.get(
                                "label"
                            ),
                            "chunk_id": edge.get(
                                "chunk_id"
                            ),
                            "quote": edge.get(
                                "quote"
                            ),
                        }
                    )

            record = {
                "candidate_id": candidate_id,
                "skills": skill_records,
                "employment": employment_records,
                "projects": project_records,
                "structured_history_available": (
                    bool(employment_records)
                    or bool(project_records)
                ),
            }

            all_evidence.append(record)

            print(
                candidate_id,
                "| skill citations:",
                len(skill_records),
                "| employment:",
                len(employment_records),
                "| projects:",
                len(project_records),
            )

        return {
            "evidence": all_evidence,
        }

    def generate(state: GraphRAGState):
        print("\n[GENERATE]")

        if not state["verified_candidates"]:
            return {
                "answer": (
                    "No candidates were verified as "
                    "mentioning all required skills. "
                    "No recommendation can be made "
                    "from the available evidence."
                )
            }

        evidence_json = json.dumps(
            state["evidence"],
            indent=2,
            ensure_ascii=False,
        )

        prompt = f"""
You are a cautious recruiting research assistant.

Answer the recruiter's question using ONLY the
structured evidence provided below.

RECRUITER QUERY:
{state["query"]}

REQUIRED SKILLS:
{state["required_skills"]}

VERIFIED CANDIDATE EVIDENCE:
{evidence_json}

STRICT RULES:

1. Every recommended candidate must appear in
   the verified evidence.

2. Keep the candidate order supplied in the evidence.

3. A skill mention is NOT proof of professional use,
   proficiency, or years of experience.

4. Never claim that a candidate used a particular
   skill in a specific job or project unless the
   evidence explicitly connects them.

5. You may describe accepted employment records
   and accepted projects, but do not infer a
   relationship between them and a required skill.

6. For every skill claim, cite its exact chunk ID.

7. For every employment or project claim, cite its
   exact chunk ID.

8. Do not invent candidate names, employers,
   job titles, dates, responsibilities, or projects.

9. If structured employment/project records are
   absent, say:
   "Structured employment/project evidence has
   not yet been extracted for this candidate."
   This does NOT mean the CV lacks experience.

10. Use concise, clear recruiter-friendly language.

11. If a source excerpt is incomplete or ambiguous,
    avoid making a stronger claim than it supports.

Return a short introduction and a numbered list
of candidates with evidence-based explanations.
"""

        response = llm.invoke(prompt)

        return {
            "answer": response.content,
        }

    workflow = StateGraph(GraphRAGState)

    workflow.add_node(
        "retrieve",
        retrieve,
    )

    workflow.add_node(
        "verify",
        verify,
    )

    workflow.add_node(
        "collect_evidence",
        collect_evidence,
    )

    workflow.add_node(
        "generate",
        generate,
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
        "collect_evidence",
    )

    workflow.add_edge(
        "collect_evidence",
        "generate",
    )

    workflow.add_edge(
        "generate",
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
        "=== LESSON 5F: GRAPH-GROUNDED ANSWERS ==="
    )

    chunks = load_chunks()
    parents = collect_candidate_chunks(chunks)
    graph = load_graph()

    print("CV chunks:", len(chunks))
    print("Candidates:", len(parents))
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    app = make_app(
        chunks,
        parents,
        graph,
    )

    result = app.invoke(
        {
            "query": args.query,
        }
    )

    print("\n" + "=" * 65)
    print("FINAL GRAPH-GROUNDED ANSWER")
    print("=" * 65)

    print(result["answer"])

    print("\n=== LESSON 5F COMPLETE ===")


if __name__ == "__main__":
    main()
