
"""Lesson 5A: Build an evidence-backed candidate knowledge graph.

Nodes:
    candidate:<candidate_id>
    skill:<canonical_skill>
    chunk:<chunk_id>

Edges:
    candidate -> skill  (HAS_SKILL)
    candidate -> chunk  (HAS_CHUNK)
    skill -> chunk      (EVIDENCED_BY)

A HAS_SKILL edge means the technology is explicitly
mentioned in an indexed CV chunk. It does NOT prove
professional experience or proficiency.

Uses only local benchmark data and deterministic
skill matching. No LLM or API calls are required.
"""

import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx

from scripts.retrieval.demo_parent_cv_verification import (
    load_chunks,
    collect_candidate_chunks,
    find_skill_evidence,
)


OUTPUT_DIR = Path("data/public_benchmark/graph")

GRAPH_PATH = OUTPUT_DIR / "candidate_graph.json"
SUMMARY_PATH = OUTPUT_DIR / "graph_summary.json"

SKILLS = [
    "Python",
    "Docker",
    "SQL",
    "Java",
    "Linux",
    "GitLab",
    "JavaScript",
]


def candidate_node_id(candidate_id):
    return f"candidate:{candidate_id}"


def skill_node_id(skill):
    return f"skill:{skill}"


def chunk_node_id(chunk_id):
    return f"chunk:{chunk_id}"


def build_graph(chunks):
    """Create nodes and evidence-backed relationships."""

    graph = nx.DiGraph()

    parents = collect_candidate_chunks(chunks)

    # 1. Add candidate nodes.
    for candidate_id in sorted(parents):
        graph.add_node(
            candidate_node_id(candidate_id),
            node_type="candidate",
            candidate_id=candidate_id,
            label=candidate_id,
        )

    # 2. Add the supported skill vocabulary.
    for skill in SKILLS:
        graph.add_node(
            skill_node_id(skill),
            node_type="skill",
            skill=skill,
            label=skill,
        )

    # 3. Add CV chunk nodes and ownership edges.
    for chunk in chunks:
        candidate_id = chunk["candidate_id"]
        chunk_id = chunk["chunk_id"]

        graph.add_node(
            chunk_node_id(chunk_id),
            node_type="chunk",
            chunk_id=chunk_id,
            candidate_id=candidate_id,
            label=chunk_id,
        )

        graph.add_edge(
            candidate_node_id(candidate_id),
            chunk_node_id(chunk_id),
            relation="HAS_CHUNK",
        )

    # 4. Add skill relationships using the same
    #    evidence matcher as the final LangGraph agent.
    for candidate_id, candidate_chunks in parents.items():
        evidence = find_skill_evidence(
            candidate_chunks,
            SKILLS,
        )

        for skill in SKILLS:
            matching_chunk_ids = evidence.get(skill, [])

            if not matching_chunk_ids:
                continue

            # Candidate -> skill.
            graph.add_edge(
                candidate_node_id(candidate_id),
                skill_node_id(skill),
                relation="HAS_SKILL",
                evidence_count=len(matching_chunk_ids),
                evidence_chunk_ids=matching_chunk_ids,
            )

            # Skill -> supporting chunks.
            for chunk_id in matching_chunk_ids:
                graph.add_edge(
                    skill_node_id(skill),
                    chunk_node_id(chunk_id),
                    relation="EVIDENCED_BY",
                )

    return graph


def summarize_graph(graph):
    """Calculate graph statistics."""

    node_types = Counter(
        data["node_type"]
        for _, data in graph.nodes(data=True)
    )

    edge_types = Counter(
        data["relation"]
        for _, _, data in graph.edges(data=True)
    )

    skill_counts = {}

    for skill in SKILLS:
        skill_id = skill_node_id(skill)

        candidates = [
            node
            for node in graph.predecessors(skill_id)
            if graph.nodes[node]["node_type"] == "candidate"
        ]

        skill_counts[skill] = len(candidates)

    return {
        "total_nodes": graph.number_of_nodes(),
        "total_edges": graph.number_of_edges(),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "candidate_counts_by_skill": skill_counts,
    }


def save_graph(graph, summary):
    """Export graph and statistics as readable JSON."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = nx.node_link_data(
        graph,
        edges="links",
    )

    GRAPH_PATH.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def print_example_query(graph):
    """Show candidates mentioning both Python and SQL."""

    python_candidates = set(
        graph.predecessors(skill_node_id("Python"))
    )

    sql_candidates = set(
        graph.predecessors(skill_node_id("SQL"))
    )

    matches = sorted(
        python_candidates & sql_candidates
    )

    print("\n=== EXAMPLE GRAPH QUERY ===")
    print("Candidates with Python AND SQL:")
    print("Total:", len(matches))

    for node in matches[:10]:
        candidate_id = graph.nodes[node]["candidate_id"]

        python_edge = graph.edges[
            node,
            skill_node_id("Python"),
        ]

        sql_edge = graph.edges[
            node,
            skill_node_id("SQL"),
        ]

        print(
            f"\nCandidate: {candidate_id}"
        )

        print(
            "  Python evidence:",
            python_edge["evidence_chunk_ids"],
        )

        print(
            "  SQL evidence:",
            sql_edge["evidence_chunk_ids"],
        )


def main():
    print("=== LESSON 5A: CANDIDATE KNOWLEDGE GRAPH ===")

    chunks = load_chunks()

    print("Loaded CV chunks:", len(chunks))

    graph = build_graph(chunks)
    summary = summarize_graph(graph)

    save_graph(graph, summary)

    print("\n=== GRAPH STATISTICS ===")
    print("Total nodes:", summary["total_nodes"])
    print("Total edges:", summary["total_edges"])

    print("\nNode types:")
    for key, value in summary["node_types"].items():
        print(f"  {key}: {value}")

    print("\nEdge types:")
    for key, value in summary["edge_types"].items():
        print(f"  {key}: {value}")

    print("\nCandidates mentioning each skill:")
    for skill, count in summary[
        "candidate_counts_by_skill"
    ].items():
        print(f"  {skill}: {count}")

    print_example_query(graph)

    print("\n=== SAVED FILES ===")
    print(GRAPH_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()

