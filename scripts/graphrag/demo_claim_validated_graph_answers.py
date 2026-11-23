
"""Lesson 5G.4: Claim-validated privacy-safe GraphRAG answers.

Reuses:
- BM25 candidate ranking
- Graph skill verification
- Original evidence registry
- Privacy-safe evidence minimization
- Deterministic semantic claim validation

The final answer is generated from approved claim templates.
No LLM-generated free-text statements are rendered.

This is a benchmark demonstration, not a production-grade
privacy or CV authenticity verification system.
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from scripts.demo_parent_cv_verification import (
    load_chunks,
    collect_candidate_chunks,
)

from scripts.evaluate_graph_hybrid_retrieval import (
    build_bm25,
    rank_candidates_bm25,
    extract_skills,
    graph_verified_candidates,
    filter_ranking,
    load_graph,
)

from scripts.graphrag.demo_deterministic_graph_answers import (
    collect_evidence_registry,
    compute_skill_coverage,
    stable_query_id,
)

from scripts.graphrag.demo_validated_graph_answers import (
    normalize_retrieval_query,
)

from scripts.graphrag.demo_privacy_safe_graph_answers import (
    make_safe_registry,
)

from scripts.validate_semantic_claims import (
    validate_claim,
)


TOP_K = 5

OUTPUT_DIR = Path(
    "data/public_benchmark/graph/claim_validated_answers"
)


def build_claims(
    selected,
    required_skills,
    safe_registry,
    candidate_map,
):
    """Create one claim per required skill per candidate.

    Only records in the privacy-safe registry are used.
    """

    proposed = []

    for candidate_id in selected:
        evidence_ids = candidate_map.get(
            candidate_id,
            [],
        )

        for skill in required_skills:
            matching_ids = []

            for evidence_id in evidence_ids:
                record = safe_registry.get(evidence_id)

                if record is None:
                    continue

                if record.get("type") != "skill":
                    continue

                if record.get("candidate_id") != candidate_id:
                    continue

                if (
                    record.get("subject", "").casefold()
                    != skill.casefold()
                ):
                    continue

                matching_ids.append(evidence_id)

            if matching_ids:
                proposed.append({
                    "candidate_id": candidate_id,
                    "claim_type": "skill_mentioned",
                    "skill": skill,
                    "evidence_id": matching_ids[0],
                })

    return proposed


def validate_all_claims(
    proposed,
    safe_registry,
    selected,
):
    """Validate claims using Lesson 5G.3."""

    accepted = []
    rejected = []

    for claim in proposed:
        validated, reason = validate_claim(
            claim,
            safe_registry,
            selected,
        )

        if validated is not None:
            accepted.append(validated)
        else:
            rejected.append({
                "claim": claim,
                "reason": reason,
            })

    return accepted, rejected


def render_answer(
    selected,
    required_skills,
    accepted,
    safe_registry,
):
    """Render only validated canonical statements."""

    if not selected:
        return (
            "No candidates were verified as mentioning "
            "all required skills."
        )

    claims_by_candidate = {}

    for claim in accepted:
        claims_by_candidate.setdefault(
            claim["candidate_id"],
            {},
        )[claim["skill"].casefold()] = claim

    lines = [
        "Candidates with graph-verified skill mentions: "
        + ", ".join(required_skills),
        "",
    ]

    for index, candidate_id in enumerate(
        selected,
        start=1,
    ):
        lines.append(
            f"{index}. Candidate {candidate_id}"
        )

        candidate_claims = claims_by_candidate.get(
            candidate_id,
            {},
        )

        for skill in required_skills:
            claim = candidate_claims.get(
                skill.casefold()
            )

            if claim is None:
                lines.append(
                    f"  {skill}: No validated claim."
                )
                continue

            evidence = safe_registry[
                claim["evidence_id"]
            ]

            lines.append(
                f"  {skill}: {claim['statement']}"
            )

            lines.append(
                f"    Source: [{claim['chunk_id']}]"
            )

            lines.append(
                "    Display excerpt: "
                + evidence["display_excerpt"]
            )

        lines.append("")

    return "\n".join(lines)


def run(query):
    load_dotenv()

    print(
        "=== LESSON 5G.4: "
        "CLAIM-VALIDATED GRAPH ANSWERS ==="
    )

    chunks = load_chunks()
    graph = load_graph()

    candidate_chunks = collect_candidate_chunks(
        chunks
    )

    chunk_lookup = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    print("CV chunks:", len(chunks))
    print("Candidates:", len(candidate_chunks))
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    required_skills = extract_skills(query)

    retrieval_query = normalize_retrieval_query(
        query
    )

    print("\n[RETRIEVE]")
    print("Retrieval query:", retrieval_query)
    print("Required skills:", required_skills)

    bm25 = build_bm25(chunks)

    ranking = rank_candidates_bm25(
        retrieval_query,
        chunks,
        bm25,
    )

    verified = graph_verified_candidates(
        graph,
        required_skills,
    )

    selected = filter_ranking(
        ranking,
        verified,
    )[:TOP_K]

    print("\n[VERIFY]")
    print("Graph-verified:", len(verified))
    print("Selected:", selected)

    print("\n[COLLECT EVIDENCE]")

    registry, candidate_map = (
        collect_evidence_registry(
            selected,
            required_skills,
            candidate_chunks,
            graph,
            chunk_lookup,
        )
    )

    coverage = compute_skill_coverage(
        selected,
        required_skills,
        registry,
    )

    print(
        "Original evidence records:",
        len(registry),
    )

    print(
        "Mandatory skill coverage:",
        f"{coverage['covered']}/{coverage['expected']}",
    )

    print("\n[MINIMIZE EVIDENCE]")

    safe_registry = make_safe_registry(
        registry,
        chunk_lookup,
    )

    print(
        "Safe evidence records:",
        len(safe_registry),
    )

    print("\n[BUILD CLAIMS]")

    proposed = build_claims(
        selected,
        required_skills,
        safe_registry,
        candidate_map,
    )

    print("Proposed claims:", len(proposed))

    print("\n[VALIDATE CLAIMS]")

    accepted, rejected = validate_all_claims(
        proposed,
        safe_registry,
        selected,
    )

    print("Accepted claims:", len(accepted))
    print("Rejected claims:", len(rejected))

    print("\n[RENDER]")

    answer = render_answer(
        selected,
        required_skills,
        accepted,
        safe_registry,
    )

    print("\n" + "=" * 65)
    print("FINAL ANSWER")
    print("=" * 65)
    print(answer)

    print("=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)

    print(
        "Mandatory skill coverage:",
        f"{coverage['covered']}/{coverage['expected']}",
    )

    print("Proposed claims:", len(proposed))
    print("Accepted claims:", len(accepted))
    print("Rejected claims:", len(rejected))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / (
        stable_query_id(query) + ".json"
    )

    output = {
        "query": query,
        "retrieval_query": retrieval_query,
        "required_skills": required_skills,
        "selected_candidates": selected,
        "skill_coverage": coverage,
        "safe_evidence_registry": safe_registry,
        "proposed_claims": proposed,
        "accepted_claims": accepted,
        "rejected_claims": rejected,
        "answer": answer,
    }

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
        "\n=== LESSON 5G.4 COMPLETE ==="
    )


if __name__ == "__main__":
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

    run(args.query)
