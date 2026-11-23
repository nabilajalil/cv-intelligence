
"""Lesson 5D: Build an enriched, evidence-backed CV graph.

Adds:
- Candidate -> Employment
- Employment -> Role
- Employment -> Employer
- Candidate -> Project

Uses only previously validated extraction results.

Project-technology relationships are deferred for review.

The original candidate_graph.json is never modified.
No LLM or API calls.
"""

import json
import hashlib
from collections import defaultdict
from pathlib import Path

import networkx as nx

from scripts.retrieval.demo_parent_cv_verification import load_chunks


BASE_GRAPH_PATH = Path(
    "data/public_benchmark/graph/candidate_graph.json"
)

VALIDATION_PATH = Path(
    "data/public_benchmark/graph/"
    "relationship_validation/report.json"
)

OUTPUT_DIR = Path(
    "data/public_benchmark/graph/enriched"
)

ENRICHED_GRAPH_PATH = (
    OUTPUT_DIR / "enriched_candidate_graph.json"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def normalize(value):
    return " ".join(value.casefold().split())


def stable_id(value):
    """Create a deterministic short identifier."""

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:16]


def load_base_graph():
    if not BASE_GRAPH_PATH.exists():
        raise FileNotFoundError(
            f"Missing base graph: {BASE_GRAPH_PATH}"
        )

    data = json.loads(
        BASE_GRAPH_PATH.read_text(encoding="utf-8")
    )

    return nx.node_link_graph(
        data,
        directed=True,
        multigraph=False,
        edges="links",
    )


def load_validation_report():
    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Missing validation report: {VALIDATION_PATH}"
        )

    return json.loads(
        VALIDATION_PATH.read_text(encoding="utf-8")
    )


def load_source_chunks():
    return {
        chunk["chunk_id"]: chunk
        for chunk in load_chunks()
    }


def citation_is_valid(
    record,
    candidate_id,
    chunk_lookup,
):
    """Recheck citation before inserting anything."""

    chunk = chunk_lookup.get(
        record.get("chunk_id")
    )

    quote = record.get("quote", "")

    return (
        chunk is not None
        and chunk["candidate_id"] == candidate_id
        and bool(quote.strip())
        and quote in chunk["text"]
    )


def original_extraction_valid(record):
    """Check the original Lesson 5C decision."""

    return record.get("valid") is True


def evidence_fields(record):
    return {
        "chunk_id": record["chunk_id"],
        "quote": record["quote"],
        "source": "llm_extraction",
    }


def get_candidate_records(candidate_report):
    """Collect records passing initial extraction validation.

    Repeated job titles are allowed here because
    individual employment nodes will preserve identity.
    """

    records = []

    for record in (
        candidate_report.get("eligible_for_review", [])
        + candidate_report.get("needs_attention", [])
    ):
        if original_extraction_valid(record):
            records.append(record)

    return records


def add_employment_records(
    graph,
    candidate_id,
    records,
    chunk_lookup,
):
    """Pair roles and employers by exact supporting quotation.

    Matching on the same chunk alone is insufficient:
    a chunk may contain multiple employment records.
    """

    role_groups = defaultdict(list)
    employer_groups = defaultdict(list)

    for record in records:
        if not citation_is_valid(
            record,
            candidate_id,
            chunk_lookup,
        ):
            continue

        key = (
            record["chunk_id"],
            record["quote"],
        )

        if record["relation"] == "HELD_ROLE":
            role_groups[key].append(record)

        elif record["relation"] == "WORKED_AT":
            employer_groups[key].append(record)

    created = 0
    skipped = 0

    for key, roles in role_groups.items():
        employers = employer_groups.get(
            key,
            [],
        )

        # Require exactly one role and one employer
        # supported by the same exact quotation.
        if len(roles) != 1 or len(employers) != 1:
            skipped += len(roles)
            continue

        role_record = roles[0]
        employer_record = employers[0]

        role_name = role_record["target"]
        employer_name = employer_record["target"]

        employment_key = (
            candidate_id
            + "|"
            + key[0]
            + "|"
            + key[1]
        )

        employment_id = (
            "employment:"
            + stable_id(employment_key)
        )

        role_id = (
            "role:"
            + normalize(role_name)
        )

        employer_id = (
            "employer:"
            + normalize(employer_name)
        )

        graph.add_node(
            employment_id,
            node_type="employment",
            candidate_id=candidate_id,
            role=role_name,
            employer=employer_name,
            label=f"{role_name} at {employer_name}",
        )

        graph.add_node(
            role_id,
            node_type="role",
            label=role_name,
        )

        graph.add_node(
            employer_id,
            node_type="employer",
            label=employer_name,
        )

        graph.add_edge(
            f"candidate:{candidate_id}",
            employment_id,
            relation="HELD_POSITION",
            **evidence_fields(role_record),
        )

        graph.add_edge(
            employment_id,
            role_id,
            relation="HAS_ROLE",
            **evidence_fields(role_record),
        )

        graph.add_edge(
            employment_id,
            employer_id,
            relation="AT_EMPLOYER",
            **evidence_fields(employer_record),
        )

        created += 1

    return created, skipped


def add_project_records(
    graph,
    candidate_id,
    records,
    chunk_lookup,
):
    """Add only valid candidate-to-project relationships."""

    created = 0

    for record in records:
        if record["relation"] != "CONTRIBUTED_TO":
            continue

        if record["source"] != candidate_id:
            continue

        if not citation_is_valid(
            record,
            candidate_id,
            chunk_lookup,
        ):
            continue

        project_name = record["target"]

        project_key = (
            candidate_id
            + "|"
            + normalize(project_name)
        )

        project_id = (
            "project:"
            + stable_id(project_key)
        )

        graph.add_node(
            project_id,
            node_type="project",
            candidate_id=candidate_id,
            label=project_name,
        )

        graph.add_edge(
            f"candidate:{candidate_id}",
            project_id,
            relation="CONTRIBUTED_TO",
            **evidence_fields(record),
        )

        created += 1

    return created


def summarize_graph(graph):
    node_counts = defaultdict(int)
    edge_counts = defaultdict(int)

    for _, data in graph.nodes(data=True):
        node_counts[
            data.get("node_type", "unknown")
        ] += 1

    for _, _, data in graph.edges(data=True):
        edge_counts[
            data.get("relation", "unknown")
        ] += 1

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "node_types": dict(sorted(node_counts.items())),
        "edge_types": dict(sorted(edge_counts.items())),
    }


def main():
    print(
        "=== LESSON 5D: ENRICHED KNOWLEDGE GRAPH ==="
    )

    graph = load_base_graph()
    report = load_validation_report()
    chunk_lookup = load_source_chunks()

    original_nodes = graph.number_of_nodes()
    original_edges = graph.number_of_edges()

    print("Original nodes:", original_nodes)
    print("Original edges:", original_edges)

    employment_count = 0
    skipped_employment = 0
    project_count = 0

    for candidate_report in report["candidates"]:
        candidate_id = candidate_report["candidate_id"]

        if f"candidate:{candidate_id}" not in graph:
            print(
                "Skipping unknown candidate:",
                candidate_id,
            )
            continue

        records = get_candidate_records(
            candidate_report
        )

        jobs_added, jobs_skipped = add_employment_records(
            graph,
            candidate_id,
            records,
            chunk_lookup,
        )

        projects_added = add_project_records(
            graph,
            candidate_id,
            records,
            chunk_lookup,
        )

        employment_count += jobs_added
        skipped_employment += jobs_skipped
        project_count += projects_added

        print(
            f"{candidate_id}: "
            f"{jobs_added} employment records, "
            f"{projects_added} projects"
        )

    summary = summarize_graph(graph)

    summary["original_nodes"] = original_nodes
    summary["original_edges"] = original_edges
    summary["employment_records_added"] = (
        employment_count
    )
    summary["employment_roles_skipped"] = (
        skipped_employment
    )
    summary["projects_added"] = project_count
    summary["project_technology_edges_added"] = 0

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ENRICHED_GRAPH_PATH.write_text(
        json.dumps(
            nx.node_link_data(
                graph,
                edges="links",
            ),
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

    print("\n" + "=" * 65)
    print("ENRICHED GRAPH SUMMARY")
    print("=" * 65)

    print("Nodes:", summary["nodes"])
    print("Edges:", summary["edges"])
    print(
        "Employment records added:",
        employment_count,
    )
    print(
        "Projects added:",
        project_count,
    )
    print(
        "Employment roles skipped:",
        skipped_employment,
    )
    print(
        "Project-technology edges added:",
        0,
    )

    print("\nNode types:")

    for name, count in summary["node_types"].items():
        print(f"  {name}: {count}")

    print("\nEdge types:")

    for name, count in summary["edge_types"].items():
        print(f"  {name}: {count}")

    print("\nSaved files:")
    print(" ", ENRICHED_GRAPH_PATH)
    print(" ", SUMMARY_PATH)

    print("\n=== LESSON 5D COMPLETE ===")


if __name__ == "__main__":
    main()
