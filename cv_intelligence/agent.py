
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from .hybrid_candidates import search_hybrid_candidates
from .candidate_evaluation import evaluate_retrieved_candidates
from .query_understanding import understand_query
from .candidate_summary import summarize_candidate

class AgentState(TypedDict):
    query: str
    required_skills: list[str]
    limit: int
    candidates: list[dict]
    evaluated_candidates: list[dict]


def understand_node(state: AgentState) -> dict:
    """Extract required skills if none were provided."""

    if state.get("required_skills"):
        return {}

    skills = understand_query(state["query"])

    return {"required_skills": skills}


def retrieve_node(state: AgentState) -> dict:
    """Retrieve distinct candidates using hybrid search."""

    candidates = search_hybrid_candidates(
        query=state["query"],
        limit=state["limit"],
    )

    return {"candidates": candidates}


def evaluate_node(state: AgentState) -> dict:
    """Evaluate candidates using indexed CV evidence."""

    skills = state["required_skills"]

    if not skills:
        return {
            "evaluated_candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "document_id": candidate["document_id"],
                    "best_rrf_score": candidate["best_rrf_score"],
                    "matched_skills": [],
                    "skills_not_found_in_evidence": [],
                    "matched_skill_count": 0,
                    "required_skill_count": 0,
                    "skill_evidence": {},
                    "evaluation_status": "no_explicit_skills",
                }
                for candidate in state["candidates"]
            ]
        }

    results = evaluate_retrieved_candidates(
        candidates=state["candidates"],
        required_skills=skills,
    )

    return {"evaluated_candidates": results}


def summarize_node(state: AgentState) -> dict:
    """Attach an evidence-grounded candidate summary."""

    summarized = []

    polishing_enabled = (
        os.getenv("CV_ENABLE_LLM_POLISHING", "false").lower()
        == "true"
    )

    for candidate in state["evaluated_candidates"]:
        if polishing_enabled:
            from .prompts.response_service import (
                polish_candidate_response,
            )

            response = polish_candidate_response(candidate)

            summarized.append({
                **candidate,
                "summary": response["summary"],
                "response_metadata": {
                    key: value
                    for key, value in response.items()
                    if key != "summary"
                },
            })
        else:
            summarized.append({
                **candidate,
                "summary": summarize_candidate(candidate),
            })

    return {"evaluated_candidates": summarized}


def build_agent():
    workflow = StateGraph(AgentState)

    workflow.add_node("understand", understand_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("summarize", summarize_node)

    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "retrieve")
    workflow.add_edge("retrieve", "evaluate")
    workflow.add_edge("evaluate", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()
