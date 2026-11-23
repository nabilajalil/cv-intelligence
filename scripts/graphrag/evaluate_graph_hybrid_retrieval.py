
"""Lesson 5E: Evaluate hybrid BM25 + knowledge graph retrieval.

Compares:
1. Chunk-level BM25
2. BM25 + full-parent skill verification
3. BM25 + graph-based skill verification

Uses the existing 50-CV benchmark and five queries.

No OpenAI calls.
No embedding API calls.
No changes to the saved knowledge graph.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from rank_bm25 import BM25Okapi

from scripts.retrieval.demo_parent_cv_verification import (
    load_chunks,
    collect_candidate_chunks,
    find_skill_evidence,
)

from scripts.evaluation.evaluate_hybrid_rrf import (
    load_queries,
    load_qrels,
    evaluate_ranking,
    mean_metrics,
    print_metrics,
)


GRAPH_PATH = Path(
    "data/public_benchmark/graph/enriched/"
    "enriched_candidate_graph.json"
)

OUTPUT_PATH = Path(
    "data/public_benchmark/graph/"
    "graph_hybrid_evaluation.json"
)

TOP_K = 5

SKILLS = [
    "Python",
    "Docker",
    "SQL",
    "Java",
    "Linux",
    "GitLab",
    "JavaScript",
]


def load_graph():
    """Load the enriched NetworkX graph."""

    if not GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing graph: {GRAPH_PATH}\n"
            "Run Lesson 5D first."
        )

    data = json.loads(
        GRAPH_PATH.read_text(encoding="utf-8")
    )

    return nx.node_link_graph(
        data,
        directed=True,
        multigraph=False,
        edges="links",
    )


def tokenize(text):
    """Tokenize text consistently for this experiment."""

    return re.findall(
        r"[a-z0-9]+",
        text.casefold(),
    )


def extract_skills(query):
    """Extract known mandatory technologies from a query."""

    found = []

    for skill in SKILLS:
        pattern = (
            r"(?<![A-Za-z0-9])"
            + re.escape(skill)
            + r"(?![A-Za-z0-9])"
        )

        if re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        ):
            found.append(skill)

    return found


def build_bm25(chunks):
    """Build BM25 over all original CV chunks."""

    tokenized = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    return BM25Okapi(tokenized)


def rank_candidates_bm25(
    query,
    chunks,
    bm25,
):
    """Rank candidates using their best chunk score."""

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    candidate_scores = defaultdict(
        lambda: float("-inf")
    )

    for chunk, score in zip(chunks, scores):
        candidate_id = chunk["candidate_id"]

        candidate_scores[candidate_id] = max(
            candidate_scores[candidate_id],
            float(score),
        )

    ranking = sorted(
        candidate_scores,
        key=lambda candidate_id: (
            -candidate_scores[candidate_id],
            candidate_id,
        ),
    )

    return ranking


def parent_verified_candidates(
    parents,
    required_skills,
):
    """Candidates with evidence for every required skill."""

    if not required_skills:
        return set(parents)

    verified = set()

    for candidate_id, candidate_chunks in parents.items():
        evidence = find_skill_evidence(
            candidate_chunks,
            required_skills,
        )

        if all(
            evidence.get(skill)
            for skill in required_skills
        ):
            verified.add(candidate_id)

    return verified


def graph_verified_candidates(
    graph,
    required_skills,
):
    """Traverse graph skill relationships and intersect sets."""

    if not required_skills:
        return {
            data["candidate_id"]
            for _, data in graph.nodes(data=True)
            if data.get("node_type") == "candidate"
        }

    matching_sets = []

    for skill in required_skills:
        skill_id = f"skill:{skill}"

        if skill_id not in graph:
            return set()

        candidates = set()

        for candidate_node in graph.predecessors(
            skill_id
        ):
            edge = graph.edges[
                candidate_node,
                skill_id,
            ]

            if edge.get("relation") != "HAS_SKILL":
                continue

            node_data = graph.nodes[
                candidate_node
            ]

            if node_data.get("node_type") != "candidate":
                continue

            candidates.add(
                node_data["candidate_id"]
            )

        matching_sets.append(candidates)

    return set.intersection(
        *matching_sets
    )


def filter_ranking(
    ranking,
    allowed_candidates,
):
    """Preserve BM25 order while enforcing eligibility."""

    return [
        candidate_id
        for candidate_id in ranking
        if candidate_id in allowed_candidates
    ]


def format_result(metrics):
    return {
        "precision_at_5": metrics["precision"],
        "recall_at_5": metrics["recall"],
        "ndcg_at_5": metrics["ndcg"],
    }


def main():
    print(
        "=== LESSON 5E: HYBRID BM25 + GRAPH EVALUATION ==="
    )

    chunks = load_chunks()
    parents = collect_candidate_chunks(chunks)
    graph = load_graph()

    queries = load_queries()
    qrels = load_qrels()

    bm25 = build_bm25(chunks)

    print("CV chunks:", len(chunks))
    print("Candidates:", len(parents))
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())
    print("Queries:", len(queries))

    bm25_rows = []
    parent_rows = []
    graph_rows = []

    query_results = []

    for query in queries:
        query_id = query["query_id"]
        query_text = query["text"]

        required_skills = extract_skills(
            query_text
        )

        bm25_ranking = rank_candidates_bm25(
            query_text,
            chunks,
            bm25,
        )

        parent_allowed = parent_verified_candidates(
            parents,
            required_skills,
        )

        graph_allowed = graph_verified_candidates(
            graph,
            required_skills,
        )

        parent_ranking = filter_ranking(
            bm25_ranking,
            parent_allowed,
        )

        graph_ranking = filter_ranking(
            bm25_ranking,
            graph_allowed,
        )

        # Both verification methods should agree because
        # the graph was built from the same skill evidence.
        same_verified_set = (
            parent_allowed == graph_allowed
        )

        same_ranking = (
            parent_ranking == graph_ranking
        )

        # Fail loudly if our graph traversal produces
        # different eligibility results.
        if not same_verified_set:
            raise AssertionError(
                f"Verification mismatch for {query_id}:\n"
                f"Parent only: "
                f"{sorted(parent_allowed - graph_allowed)}\n"
                f"Graph only: "
                f"{sorted(graph_allowed - parent_allowed)}"
            )

        grades = qrels[query_id]

        bm25_metrics = evaluate_ranking(
            bm25_ranking,
            grades,
        )

        parent_metrics = evaluate_ranking(
            parent_ranking,
            grades,
        )

        graph_metrics = evaluate_ranking(
            graph_ranking,
            grades,
        )

        bm25_rows.append(bm25_metrics)
        parent_rows.append(parent_metrics)
        graph_rows.append(graph_metrics)

        query_results.append(
            {
                "query_id": query_id,
                "query": query_text,
                "required_skills": required_skills,
                "parent_verified_count": len(
                    parent_allowed
                ),
                "graph_verified_count": len(
                    graph_allowed
                ),
                "same_verified_set": same_verified_set,
                "same_ranking": same_ranking,
                "bm25_top_5": bm25_ranking[:TOP_K],
                "parent_top_5": parent_ranking[:TOP_K],
                "graph_top_5": graph_ranking[:TOP_K],
                "bm25_metrics": format_result(
                    bm25_metrics
                ),
                "parent_metrics": format_result(
                    parent_metrics
                ),
                "graph_metrics": format_result(
                    graph_metrics
                ),
            }
        )

        print("\n" + "-" * 65)
        print(
            f"{query_id}: {query_text}"
        )

        print(
            "Required skills:",
            required_skills,
        )

        print(
            "Parent verified:",
            len(parent_allowed),
        )

        print(
            "Graph verified:",
            len(graph_allowed),
        )

        print(
            "Verification sets match:",
            same_verified_set,
        )

        print(
            "Rankings match:",
            same_ranking,
        )

        print(
            "BM25 top 5:",
            bm25_ranking[:TOP_K],
        )

        print(
            "Graph top 5:",
            graph_ranking[:TOP_K],
        )

    bm25_mean = mean_metrics(bm25_rows)
    parent_mean = mean_metrics(parent_rows)
    graph_mean = mean_metrics(graph_rows)

    print("\n" + "=" * 65)
    print("MEAN RETRIEVAL METRICS")
    print("=" * 65)

    print_metrics(
        "BM25",
        bm25_mean,
    )

    print_metrics(
        "BM25 + Parent",
        parent_mean,
    )

    print_metrics(
        "BM25 + Graph",
        graph_mean,
    )

    output = {
        "benchmark": {
            "candidate_count": len(parents),
            "chunk_count": len(chunks),
            "query_count": len(queries),
            "top_k": TOP_K,
        },
        "methods": {
            "bm25": format_result(bm25_mean),
            "bm25_parent": format_result(
                parent_mean
            ),
            "bm25_graph": format_result(
                graph_mean
            ),
        },
        "queries": query_results,
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

    print("\nSaved results:", OUTPUT_PATH)
    print("\n=== LESSON 5E COMPLETE ===")


if __name__ == "__main__":
    main()
