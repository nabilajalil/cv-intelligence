
"""Lesson 5B: Query and traverse the CV knowledge graph.

Demonstrates:
1. Candidate filtering by required skills
2. Candidate -> skill -> candidate traversal
3. Candidate -> skill -> evidence chunk traversal
4. Cross-checking graph relationships against source CV text

No LLM or external API calls.
"""

import json
from collections import defaultdict
from pathlib import Path

import networkx as nx

from scripts.demo_parent_cv_verification import load_chunks


GRAPH_PATH = Path(
    "data/public_benchmark/graph/candidate_graph.json"
)

REQUIRED_SKILLS = ["Python", "SQL"]
REFERENCE_CANDIDATE = "r00041"

TOP_SIMILAR = 5
MAX_EVIDENCE_CHARS = 350


def load_graph():
    """Load the directed graph created in Lesson 5A."""

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Graph not found: {GRAPH_PATH}\n"
            "Run Lesson 5A first:\n"
            "python -m scripts.graphrag.build_candidate_knowledge_graph"
        )

    data = json.loads(
        GRAPH_PATH.read_text(encoding="utf-8")
    )

    graph = nx.node_link_graph(
        data,
        directed=True,
        multigraph=False,
        edges="links",
    )

    return graph


def candidate_node(candidate_id):
    return f"candidate:{candidate_id}"


def skill_node(skill):
    return f"skill:{skill}"


def chunk_node(chunk_id):
    return f"chunk:{chunk_id}"


def candidate_id_from_node(graph, node):
    return graph.nodes[node]["candidate_id"]


def candidates_with_all_skills(graph, required_skills):
    """Find the intersection of candidate sets."""

    candidate_sets = []

    for skill in required_skills:
        node = skill_node(skill)

        if node not in graph:
            return []

        matching_candidates = {
            predecessor
            for predecessor in graph.predecessors(node)
            if graph.nodes[predecessor].get("node_type")
            == "candidate"
        }

        candidate_sets.append(matching_candidates)

    if not candidate_sets:
        return []

    common = set.intersection(*candidate_sets)

    return sorted(
        candidate_id_from_node(graph, node)
        for node in common
    )


def skills_for_candidate(graph, candidate_id):
    """Traverse outgoing HAS_SKILL relationships."""

    node = candidate_node(candidate_id)

    if node not in graph:
        raise ValueError(
            f"Candidate not found: {candidate_id}"
        )

    skills = set()

    for neighbor in graph.successors(node):
        edge = graph.edges[node, neighbor]

        if edge.get("relation") == "HAS_SKILL":
            skills.add(graph.nodes[neighbor]["skill"])

    return skills


def similar_candidates(graph, reference_candidate):
    """Find other candidates through shared skill nodes.

    Traversal:
        reference candidate -> skill <- other candidate

    The score is the number of shared skills.
    """

    reference_skills = skills_for_candidate(
        graph,
        reference_candidate,
    )

    shared = defaultdict(set)

    for skill in reference_skills:
        skill_id = skill_node(skill)

        for other_node in graph.predecessors(skill_id):
            if graph.nodes[other_node].get("node_type") != "candidate":
                continue

            other_id = candidate_id_from_node(
                graph,
                other_node,
            )

            if other_id == reference_candidate:
                continue

            shared[other_id].add(skill)

    results = []

    for other_id, common_skills in shared.items():
        results.append(
            {
                "candidate_id": other_id,
                "shared_skills": sorted(common_skills),
                "shared_count": len(common_skills),
            }
        )

    results.sort(
        key=lambda item: (
            -item["shared_count"],
            item["candidate_id"],
        )
    )

    return results


def evidence_for_candidate_skill(
    graph,
    candidate_id,
    skill,
    chunk_lookup,
):
    """Traverse candidate -> skill -> supporting chunks.

    Only return chunks owned by the requested candidate.
    """

    c_node = candidate_node(candidate_id)
    s_node = skill_node(skill)

    if not graph.has_edge(c_node, s_node):
        return []

    relationship = graph.edges[c_node, s_node]

    evidence_ids = relationship.get(
        "evidence_chunk_ids",
        [],
    )

    results = []

    for chunk_id in evidence_ids:
        evidence_node = chunk_node(chunk_id)

        # Verify the supporting relationship exists.
        if not graph.has_edge(s_node, evidence_node):
            continue

        # Verify the candidate owns this chunk.
        if not graph.has_edge(c_node, evidence_node):
            continue

        chunk = chunk_lookup.get(chunk_id)

        if chunk is None:
            continue

        if chunk["candidate_id"] != candidate_id:
            continue

        results.append(
            {
                "chunk_id": chunk_id,
                "text": chunk["text"],
            }
        )

    return results


def print_skill_intersection(graph):
    print("\n" + "=" * 65)
    print("QUERY 1: CANDIDATES WITH ALL REQUIRED SKILLS")
    print("=" * 65)

    matches = candidates_with_all_skills(
        graph,
        REQUIRED_SKILLS,
    )

    print("Required skills:", REQUIRED_SKILLS)
    print("Matching candidates:", len(matches))

    for candidate_id in matches:
        print(" ", candidate_id)


def print_candidate_similarity(graph):
    print("\n" + "=" * 65)
    print("QUERY 2: MULTI-HOP CANDIDATE SIMILARITY")
    print("=" * 65)

    reference_skills = skills_for_candidate(
        graph,
        REFERENCE_CANDIDATE,
    )

    print("Reference candidate:", REFERENCE_CANDIDATE)
    print("Known skills:", sorted(reference_skills))

    results = similar_candidates(
        graph,
        REFERENCE_CANDIDATE,
    )

    print("\nTop similar candidates:")

    for result in results[:TOP_SIMILAR]:
        print(
            f"  {result['candidate_id']}: "
            f"{result['shared_count']} shared skills "
            f"{result['shared_skills']}"
        )


def print_evidence_trace(graph, chunk_lookup):
    print("\n" + "=" * 65)
    print("QUERY 3: EVIDENCE TRAVERSAL")
    print("=" * 65)

    candidate_id = REFERENCE_CANDIDATE

    print("Candidate:", candidate_id)

    for skill in REQUIRED_SKILLS:
        print(f"\nSkill: {skill}")

        evidence = evidence_for_candidate_skill(
            graph,
            candidate_id,
            skill,
            chunk_lookup,
        )

        if not evidence:
            print("  No supporting graph evidence.")
            continue

        for item in evidence:
            excerpt = " ".join(
                item["text"].split()
            )

            excerpt = excerpt[:MAX_EVIDENCE_CHARS]

            print("  Chunk:", item["chunk_id"])
            print("  Text:", excerpt)

            if len(item["text"]) > MAX_EVIDENCE_CHARS:
                print("  ...")


def main():
    print("=== LESSON 5B: GRAPH QUERY AND TRAVERSAL ===")

    graph = load_graph()

    chunks = load_chunks()

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())
    print("Source chunks:", len(chunk_lookup))

    print_skill_intersection(graph)

    print_candidate_similarity(graph)

    print_evidence_trace(graph, chunk_lookup)

    print("\n=== LESSON 5B COMPLETE ===")


if __name__ == "__main__":
    main()
