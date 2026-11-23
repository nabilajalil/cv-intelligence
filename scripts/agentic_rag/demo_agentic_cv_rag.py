
"""Lesson 4A: Agentic RAG with LangGraph for CV Intelligence.

Workflow:
RETRIEVE -> GRADE -> ROUTE
                  -> GENERATE
                  -> REWRITE -> RETRIEVE
                  -> FALLBACK
"""

import json
import re
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi


CHUNKS_PATH = Path("data/public_benchmark/chunks.jsonl")
MODEL = "gpt-4.1-mini"
MAX_RETRIES = 2
TOP_K = 5


class RAGState(TypedDict):
    query: str
    rewritten_query: str
    documents: list[dict]
    generation: str
    relevance_score: float
    retry_count: int
    max_retries: int


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def load_chunks():
    with CHUNKS_PATH.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def build_graph(chunks):
    """Create an agent with a reusable BM25 index."""

    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
    )

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(state: RAGState):
        search_query = (
            state.get("rewritten_query")
            or state["query"]
        )

        print(f"\n[RETRIEVE] {search_query}")

        scores = bm25.get_scores(
            tokenize(search_query)
        )

        ranked_indices = sorted(
            range(len(chunks)),
            key=lambda i: (-scores[i], i),
        )

        # Return at most one matching chunk per CV.
        documents = []
        seen_candidates = set()

        for index in ranked_indices:
            if scores[index] <= 0:
                break

            chunk = chunks[index]
            candidate_id = chunk["candidate_id"]

            if candidate_id in seen_candidates:
                continue

            seen_candidates.add(candidate_id)

            documents.append({
                "candidate_id": candidate_id,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "bm25_score": float(scores[index]),
            })

            if len(documents) >= TOP_K:
                break

        print(
            f"[RETRIEVE] Found {len(documents)} candidates"
        )

        return {"documents": documents}

    def grade(state: RAGState):
        print("\n[GRADE] Checking relevance...")

        relevant_documents = []
        scores = []

        for document in state["documents"]:
            response = llm.invoke([
                (
                    "system",
                    "You evaluate evidence from candidate CVs. "
                    "Score relevance to the ORIGINAL recruiter "
                    "request. Respond with only one number: "
                    "0.0, 0.3, 0.7, or 1.0. "
                    "1.0 = strong direct evidence; "
                    "0.7 = useful evidence; "
                    "0.3 = weak or partial evidence; "
                    "0.0 = unrelated. "
                    "Do not infer skills not stated in the CV."
                ),
                (
                    "human",
                    f"Recruiter request:\n{state['query']}\n\n"
                    f"CV evidence:\n{document['text']}"
                ),
            ])

            try:
                score = float(
                    str(response.content).strip()
                )
                score = max(0.0, min(1.0, score))
            except (ValueError, TypeError):
                score = 0.0

            scores.append(score)

            print(
                f"  {document['candidate_id']}: "
                f"{score:.2f}"
            )

            if score >= 0.7:
                relevant_documents.append({
                    **document,
                    "relevance_score": score,
                })

        average = (
            sum(scores) / len(scores)
            if scores else 0.0
        )

        print(
            f"[GRADE] Kept "
            f"{len(relevant_documents)}/"
            f"{len(state['documents'])}"
        )

        return {
            "documents": relevant_documents,
            "relevance_score": average,
        }

    def rewrite(state: RAGState):
        attempt = state["retry_count"] + 1

        print(f"\n[REWRITE] Attempt {attempt}")

        response = llm.invoke([
            (
                "system",
                "Rewrite the recruiter's search query "
                "to improve CV keyword retrieval. "
                "Preserve all mandatory skills and "
                "constraints. You may add synonyms, "
                "but do not replace required technologies "
                "with unrelated ones. "
                "Return only the rewritten search query."
            ),
            (
                "human",
                f"Original request:\n{state['query']}\n\n"
                f"Previous search:\n"
                f"{state.get('rewritten_query') or state['query']}"
            ),
        ])

        rewritten = str(response.content).strip()

        if not rewritten:
            rewritten = state["query"]

        print(f"[REWRITE] New query: {rewritten}")

        return {
            "rewritten_query": rewritten,
            "retry_count": attempt,
        }

    def generate(state: RAGState):
        print("\n[GENERATE] Preparing answer...")

        evidence = "\n\n".join(
            f"Candidate: {doc['candidate_id']}\n"
            f"Chunk: {doc['chunk_id']}\n"
            f"Evidence: {doc['text']}"
            for doc in state["documents"]
        )

        response = llm.invoke([
            (
                "system",
                "You are a recruiting research assistant. "
                "Answer using ONLY the provided CV evidence. "
                "Cite candidate IDs for each recommendation. "
                "Distinguish demonstrated skills from "
                "uncertain or missing information. "
                "Do not invent qualifications. "
                "A retrieved chunk is partial CV evidence, "
                "not proof that the candidate lacks other skills."
            ),
            (
                "human",
                f"Recruiter request:\n{state['query']}\n\n"
                f"Retrieved evidence:\n{evidence}"
            ),
        ])

        return {
            "generation": str(response.content)
        }

    def fallback(state: RAGState):
        print("\n[FALLBACK] No sufficient evidence")

        return {
            "generation": (
                "I couldn't find sufficiently relevant "
                "CV evidence for this recruiter request: "
                f"{state['query']}\n"
                "Try different requirements or expand "
                "the candidate collection."
            )
        }

    def route(
        state: RAGState,
    ) -> Literal["generate", "rewrite", "fallback"]:

        if state["documents"]:
            print("[ROUTER] -> GENERATE")
            return "generate"

        if state["retry_count"] < state["max_retries"]:
            print("[ROUTER] -> REWRITE")
            return "rewrite"

        print("[ROUTER] -> FALLBACK")
        return "fallback"

    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade", grade)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("fallback", fallback)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "grade")

    workflow.add_conditional_edges(
        "grade",
        route,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


def main():
    load_dotenv()

    chunks = load_chunks()

    print("=== LESSON 4A: AGENTIC CV RAG ===")
    print("Indexed chunks:", len(chunks))

    app = build_graph(chunks)

    import sys

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Find candidates with Python and Docker"
    )

    initial_state = {
        "query": query,
        "rewritten_query": "",
        "documents": [],
        "generation": "",
        "relevance_score": 0.0,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
    }

    result = app.invoke(
        initial_state,
        config={"recursion_limit": 15},
    )

    print("\n=== FINAL ANSWER ===")
    print(result["generation"])

    print("\n=== WORKFLOW SUMMARY ===")
    print("Original query:", result["query"])
    print(
        "Final search query:",
        result["rewritten_query"] or result["query"],
    )
    print("Retries:", result["retry_count"])
    print("Relevant CVs:", len(result["documents"]))


if __name__ == "__main__":
    main()
