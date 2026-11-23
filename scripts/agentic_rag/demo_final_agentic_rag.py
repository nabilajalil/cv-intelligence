
"""Lesson 4J: Final Agentic CV RAG integration.

Combines:
- BM25 candidate retrieval
- Full parent-CV mandatory skill verification
- LangGraph conditional routing
- Query rewriting
- Grounded answer generation
- Honest fallback

Does not use unstable LLM evidence scores for ranking.
Does not modify existing project scripts.
"""

import re
import sys
from collections import defaultdict
from typing import TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from rank_bm25 import BM25Okapi

from scripts.demo_parent_cv_verification import (
    collect_candidate_chunks,
    extract_required_skills,
    find_skill_evidence,
    load_chunks,
)


TOP_K = 5
CANDIDATE_POOL = 50
MAX_RETRIES = 2

MODEL = "gpt-4.1-mini"


class AgentState(TypedDict):
    query: str
    search_query: str
    required_skills: list[str]
    retrieved_candidates: list[str]
    verified_candidates: list[dict]
    answer: str
    retry_count: int
    max_retries: int


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def build_retrieval_index(chunks):
    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    parents = collect_candidate_chunks(chunks)

    return bm25, parents


def retrieve_node(state: AgentState) -> dict:
    print("\n[RETRIEVE] Query:", state["search_query"])

    scores = BM25_INDEX.get_scores(
        tokenize(state["search_query"])
    )

    candidate_scores = {}

    for chunk, score in zip(CHUNKS, scores):
        candidate_id = chunk["candidate_id"]
        score = float(score)

        if (
            candidate_id not in candidate_scores
            or score > candidate_scores[candidate_id]
        ):
            candidate_scores[candidate_id] = score

    ranked = sorted(
        candidate_scores,
        key=lambda candidate_id: (
            -candidate_scores[candidate_id],
            candidate_id,
        ),
    )

    # Ignore candidates without a positive retrieval score.
    ranked = [
        candidate_id
        for candidate_id in ranked
        if candidate_scores[candidate_id] > 0
    ]

    selected = ranked[:CANDIDATE_POOL]

    print(
        "[RETRIEVE] Candidates with positive scores:",
        len(selected),
    )

    return {
        "retrieved_candidates": selected,
    }


def verify_node(state: AgentState) -> dict:
    print("\n[VERIFY] Checking parent CV evidence...")

    verified = []

    for candidate_id in state["retrieved_candidates"]:
        candidate_chunks = PARENTS[candidate_id]

        evidence = find_skill_evidence(
            candidate_chunks,
            state["required_skills"],
        )

        full_match = all(
            evidence.get(skill)
            for skill in state["required_skills"]
        )

        if not full_match:
            continue

        verified.append({
            "candidate_id": candidate_id,
            "skill_evidence": evidence,
        })

        if len(verified) >= TOP_K:
            break

    print(
        "[VERIFY] Full matches selected:",
        len(verified),
    )

    for candidate in verified:
        print(
            " ",
            candidate["candidate_id"],
            candidate["skill_evidence"],
        )

    return {
        "verified_candidates": verified,
    }


def route_after_verification(state: AgentState) -> str:
    if state["verified_candidates"]:
        print("[ROUTER] -> GENERATE")
        return "generate"

    if state["retry_count"] < state["max_retries"]:
        print("[ROUTER] -> REWRITE")
        return "rewrite"

    print("[ROUTER] -> FALLBACK")
    return "fallback"


def rewrite_node(state: AgentState) -> dict:
    print("\n[REWRITE] Improving retrieval query...")

    skills = state["required_skills"]

    response = LLM.invoke([
        (
            "system",
            """
Rewrite the recruiter's search query for keyword
retrieval over CVs.

Rules:
- Preserve all mandatory technologies exactly.
- Do not introduce new mandatory requirements.
- Use plain-language keywords, not Boolean syntax.
- Return one short search query only.
"""
        ),
        (
            "human",
            f"Original query: {state['query']}\n"
            f"Mandatory skills: {', '.join(skills)}\n"
            f"Previous search: {state['search_query']}"
        ),
    ])

    rewritten = response.content.strip()

    # Do not allow a rewrite to drop a mandatory skill.
    missing = [
        skill
        for skill in skills
        if not re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(skill)}"
            rf"(?![A-Za-z0-9_])",
            rewritten,
            flags=re.IGNORECASE,
        )
    ]

    if missing:
        rewritten = (
            " ".join(skills)
            + " professional experience projects"
        )

    print("[REWRITE] New query:", rewritten)

    return {
        "search_query": rewritten,
        "retry_count": state["retry_count"] + 1,
    }


def build_evidence_context(candidates, required_skills):
    """Include only cited chunks that mention required skills."""

    blocks = []

    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        evidence = candidate["skill_evidence"]

        needed_ids = set()

        for skill in required_skills:
            needed_ids.update(evidence.get(skill, []))

        cited_chunks = [
            chunk
            for chunk in PARENTS[candidate_id]
            if chunk["chunk_id"] in needed_ids
        ]

        chunk_text = "\n\n".join(
            f"[{chunk['chunk_id']}]\n{chunk['text']}"
            for chunk in cited_chunks
        )

        blocks.append(
            f"CANDIDATE: {candidate_id}\n"
            f"SKILL EVIDENCE: {evidence}\n"
            f"CV CHUNKS:\n{chunk_text}"
        )

    return "\n\n---\n\n".join(blocks)


def generate_node(state: AgentState) -> dict:
    print("\n[GENERATE] Writing grounded answer...")

    context = build_evidence_context(
        state["verified_candidates"],
        state["required_skills"],
    )

    response = LLM.invoke([
        (
            "system",
            """
You are an evidence-grounded recruiting research
assistant.

Use ONLY the supplied CV evidence.

Rules:
- Include only supplied verified candidate IDs.
- Distinguish explicit skill mentions from
  demonstrated professional experience.
- Do not claim years of experience unless shown.
- Do not invent employers, projects, or proficiency.
- Cite exact chunk IDs for every skill claim.
- Do not invent chunk IDs.
- Clearly state when the evidence only shows
  a technology listed as a skill.
- Do not make final hiring recommendations.
- If the CV evidence is ambiguous, say so.
"""
        ),
        (
            "human",
            f"Recruiter request: {state['query']}\n"
            f"Mandatory skills: "
            f"{', '.join(state['required_skills'])}\n\n"
            f"Verified CV evidence:\n{context}\n\n"
            "Summarize the matching candidates with "
            "supporting chunk citations."
        ),
    ])

    return {
        "answer": response.content,
    }


def fallback_node(state: AgentState) -> dict:
    print("\n[FALLBACK] No fully verified candidates.")

    return {
        "answer": (
            "No candidates could be verified as explicitly "
            "mentioning every required technology in the "
            "indexed CV evidence. This does not prove that "
            "no suitable candidates exist. Consider "
            "adjusting the requirements or expanding "
            "the CV collection."
        ),
    }


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("verify", verify_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "verify")

    graph.add_conditional_edges(
        "verify",
        route_after_verification,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )

    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()


def main():
    global CHUNKS, BM25_INDEX, PARENTS, LLM

    load_dotenv()

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Find candidates with Python and SQL"
    )

    required_skills = extract_required_skills(query)

    print("=== LESSON 4J: FINAL AGENTIC CV RAG ===")
    print("Query:", query)
    print("Required skills:", required_skills)

    if not required_skills:
        print(
            "No supported mandatory technologies detected. "
            "Try a query containing Python, SQL, Docker, "
            "Java, Linux, GitLab, or JavaScript."
        )
        return

    CHUNKS = load_chunks()
    BM25_INDEX, PARENTS = build_retrieval_index(
        CHUNKS
    )

    LLM = ChatOpenAI(
        model=MODEL,
        temperature=0,
    )

    print("Indexed chunks:", len(CHUNKS))
    print("Candidate CVs:", len(PARENTS))

    app = build_graph()

    initial_state: AgentState = {
        "query": query,
        "search_query": query,
        "required_skills": required_skills,
        "retrieved_candidates": [],
        "verified_candidates": [],
        "answer": "",
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
    }

    result = app.invoke(initial_state)

    print("\n" + "=" * 65)
    print("FINAL ANSWER")
    print("=" * 65)

    print(result["answer"])

    print("\n=== EXECUTION SUMMARY ===")
    print("Retries:", result["retry_count"])
    print(
        "Verified candidates:",
        len(result["verified_candidates"]),
    )

    print(
        "Candidate IDs:",
        [
            candidate["candidate_id"]
            for candidate in result["verified_candidates"]
        ],
    )


if __name__ == "__main__":
    main()

