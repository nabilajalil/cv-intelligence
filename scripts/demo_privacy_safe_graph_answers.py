
"""Lesson 5G.2: Privacy-minimized GraphRAG answers.

Reuses Lesson 5F.3's deterministic evidence registry.

Original evidence remains available internally for validation.
The LLM receives only evidence IDs and minimized skill excerpts,
not the full original source quotations.

This is a benchmark demonstration, not production-grade
PII anonymization or semantic claim verification.
"""

import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from scripts.graphrag.demo_deterministic_graph_answers import (
    ExplanationBatch,
    collect_evidence_registry,
    compute_skill_coverage,
    stable_query_id,
    validate_explanations,
)

from scripts.graphrag.demo_validated_graph_answers import (
    normalize_retrieval_query,
)

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

from scripts.privacy_safe_evidence import (
    make_display_excerpt,
    EMAIL_PATTERN,
    PHONE_PATTERN,
)


OUTPUT_DIR = Path(
    "data/public_benchmark/graph/privacy_safe_answers"
)

TOP_K = 5


def make_safe_registry(registry, chunk_lookup):
    """Build a separate display registry.

    Do not modify the original evidence registry.
    """

    safe = {}

    for evidence_id, record in registry.items():
        safe_record = {
            "evidence_id": evidence_id,
            "candidate_id": record["candidate_id"],
            "type": record["type"],
            "subject": record["subject"],
            "chunk_id": record["chunk_id"],
        }

        if record["type"] == "skill":
            chunk = chunk_lookup.get(
                record["chunk_id"]
            )

            excerpt = None

            if chunk is not None:
                excerpt = make_display_excerpt(
                    chunk["text"],
                    record["subject"],
                )

            if excerpt is None:
                excerpt = (
                    "Technology mentioned: "
                    + record["subject"]
                )

            # Fail closed on detectable contact information.
            if (
                EMAIL_PATTERN.search(excerpt)
                or PHONE_PATTERN.search(excerpt)
            ):
                excerpt = (
                    "Technology mentioned: "
                    + record["subject"]
                )

            safe_record["display_excerpt"] = excerpt

        else:
            # Employment/project source quotations are
            # deliberately withheld from the LLM and
            # public-facing output in this lesson.
            safe_record["display_excerpt"] = None

        safe[evidence_id] = safe_record

    return safe


def safe_llm_payload(safe_registry, candidate_map):
    """Send only the minimum evidence metadata.

    For now, exclude employment and project evidence
    from LLM-generated explanations because their
    extracted labels may include personal information.
    """

    payload = {}

    for candidate_id, evidence_ids in candidate_map.items():
        entries = []

        for evidence_id in evidence_ids:
            record = safe_registry.get(
                evidence_id
            )

            if record is None:
                continue

            if record["type"] != "skill":
                continue

            entries.append({
                "evidence_id": evidence_id,
                "skill": record["subject"],
                "excerpt": record["display_excerpt"],
            })

        payload[candidate_id] = entries

    return payload


def validate_safe_explanations(
    explanations,
    selected,
    safe_registry,
):
    """Validate references against the safe registry.

    The LLM may reference only skill evidence exposed
    in the privacy-minimized payload.
    """

    allowed = {
        evidence_id: record
        for evidence_id, record in safe_registry.items()
        if record["type"] == "skill"
    }

    return validate_explanations(
        explanations,
        selected,
        allowed,
    )


def render_safe_answer(
    selected,
    required_skills,
    safe_registry,
    candidate_map,
    accepted,
):
    """Render only privacy-minimized evidence.

    Avoid printing original source quotations or
    unreviewed employment/project labels.
    """

    if not selected:
        return (
            "No candidates were verified as mentioning "
            "all required skills."
        )

    lines = [
        "Candidates verified for: "
        + ", ".join(required_skills),
        "",
    ]

    explanations_by_candidate = {}

    for item in accepted:
        explanations_by_candidate.setdefault(
            item["candidate_id"],
            [],
        ).append(item)

    for index, candidate_id in enumerate(
        selected,
        start=1,
    ):
        lines.append(
            f"{index}. Candidate {candidate_id}"
        )

        lines.append(
            "  Verified skill mentions:"
        )

        for skill in required_skills:
            matching = []

            for evidence_id in candidate_map.get(
                candidate_id,
                [],
            ):
                record = safe_registry.get(
                    evidence_id
                )

                if (
                    record
                    and record["type"] == "skill"
                    and record["subject"].lower()
                    == skill.lower()
                ):
                    matching.append(record)

            if not matching:
                lines.append(
                    f"    {skill}: NO EVIDENCE"
                )
                continue

            record = matching[0]

            lines.append(
                f"    {skill}: "
                f"[{record['chunk_id']}]"
            )

            lines.append(
                "      Display excerpt: "
                + record["display_excerpt"]
            )

        candidate_explanations = (
            explanations_by_candidate.get(
                candidate_id,
                [],
            )
        )

        if candidate_explanations:
            lines.append(
                "  LLM explanation "
                "(references checked; meaning not verified):"
            )

            for item in candidate_explanations:
                lines.append(
                    "    " + item["statement"]
                )

        else:
            lines.append(
                "  No validated LLM explanation."
            )

        lines.append("")

    return "\n".join(lines)


def run(query):
    load_dotenv()

    print(
        "=== LESSON 5G.2: "
        "PRIVACY-SAFE GRAPH ANSWERS ==="
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

    # This function belongs to the Lesson 5F.3
    # implementation. It retains original quotations.
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

    payload = safe_llm_payload(
        safe_registry,
        candidate_map,
    )

    print(
        "Safe evidence records:",
        len(safe_registry),
    )

    print(
        "LLM skill records:",
        sum(len(v) for v in payload.values()),
    )

    print("\n[EXPLAIN]")

    explanations = []

    if selected:
        llm = ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0,
        ).with_structured_output(
            ExplanationBatch
        )

        result = llm.invoke([
            (
                "system",
                "You write cautious candidate summaries. "
                "Use only the supplied skill evidence. "
                "A listed skill is not proof of professional "
                "experience. Do not infer proficiency, "
                "seniority, or project usage. "
                "Return one explanation per candidate. "
                "Reference only provided evidence IDs."
            ),
            (
                "human",
                json.dumps({
                    "query": retrieval_query,
                    "candidates": payload,
                }),
            ),
        ])

        explanations = result.explanations

    print("\n[VALIDATE REFERENCES]")

    accepted, rejected = (
        validate_safe_explanations(
            explanations,
            selected,
            safe_registry,
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

    print("\n[RENDER]")

    answer = render_safe_answer(
        selected,
        required_skills,
        safe_registry,
        candidate_map,
        accepted,
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

    print(
        "Accepted explanations:",
        len(accepted),
    )

    print(
        "Rejected explanations:",
        len(rejected),
    )

    # Save the safe registry and outputs only.
    # Do not duplicate original CV quotations here.
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
        "accepted_explanations": accepted,
        "rejected_explanations": rejected,
        "answer": answer,
    }

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            default=lambda obj: (
                obj.model_dump()
                if hasattr(obj, "model_dump")
                else str(obj)
            ),
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output_path)

    print(
        "\n=== LESSON 5G.2 COMPLETE ==="
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
