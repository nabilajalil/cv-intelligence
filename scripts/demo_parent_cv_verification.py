
"""Lesson 4E: Agentic RAG with parent CV evidence verification.

Pipeline:
    BM25 retrieve candidate IDs
    -> collect ALL chunks for each candidate
    -> verify mandatory skills across chunks
    -> generate / rewrite / fallback

Uses the existing 224 public benchmark chunks.
Does not modify the production CV Intelligence application.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi


CHUNKS_PATH = Path("data/public_benchmark/chunks.jsonl")

MODEL = "gpt-4.1-mini"
TOP_K = 5
CANDIDATE_POOL = 50
MAX_RETRIES = 2

# Explicitly supported benchmark technologies.
SKILLS = (
    "Python",
    "Docker",
    "SQL",
    "Java",
    "Linux",
    "GitLab",
    "JavaScript",
)


class RAGState(TypedDict):
    query: str
    rewritten_query: str
    required_skills: list[str]
    documents: list[dict]
    generation: str
    retry_count: int
    max_retries: int


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def skill_pattern(skill: str) -> re.Pattern:
    # Word boundaries prevent:
    # Java matching JavaScript
    # SQL matching NoSQL
    # Python matching longer identifiers
    return re.compile(
        rf"(?<![A-Za-z0-9_]){re.escape(skill)}"
        rf"(?![A-Za-z0-9_])",
        flags=re.IGNORECASE,
    )


def extract_required_skills(query: str) -> list[str]:
    """Detect explicit benchmark skills in the original query."""

    found = []

    for skill in SKILLS:
        if skill_pattern(skill).search(query):
            found.append(skill)

    return found


def collect_candidate_chunks(chunks: list[dict]) -> dict:
    grouped = defaultdict(list)

    for chunk in chunks:
        grouped[chunk["candidate_id"]].append(chunk)

    for candidate_id in grouped:
        grouped[candidate_id].sort(
            key=lambda chunk: chunk["chunk_index"]
        )

    return dict(grouped)


def find_skill_evidence(
    candidate_chunks: list[dict],
    required_skills: list[str],
) -> dict:
    """Find explicit skill mentions across all candidate chunks."""

    evidence = {}

    for skill in required_skills:
        pattern = skill_pattern(skill)

        matching_chunks = [
            chunk["chunk_id"]
            for chunk in candidate_chunks
            if pattern.search(chunk["text"])
        ]

        evidence[skill] = matching_chunks

    return evidence


def classify_match(evidence: dict) -> str:
    matched_count = sum(
        bool(chunk_ids)
        for chunk_ids in evidence.values()
    )

    if matched_count == len(evidence) and evidence:
        return "FULL MATCH"

    if matched_count > 0:
        return "PARTIAL MATCH"

    return "NO MATCH"


def build_graph(chunks: list[dict]):
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
    )

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    parents = collect_candidate_chunks(chunks)

    def retrieve(state: RAGState):
        search_query = (
            state.get("rewritten_query")
            or state["query"]
        )

        print(f"\n[RETRIEVE] {search_query}")

        scores = bm25.get_scores(
            tokenize(search_query)
        )

        # Rank candidates using their highest-scoring chunk.
        best_candidate_score = {}

        for index, chunk in enumerate(chunks):
            candidate_id = chunk["candidate_id"]
            score = float(scores[index])

            if (
                candidate_id not in best_candidate_score
                or score > best_candidate_score[candidate_id]
            ):
                best_candidate_score[candidate_id] = score

        ranked_candidates = sorted(
            best_candidate_score,
            key=lambda candidate_id: (
                -best_candidate_score[candidate_id],
                candidate_id,
            ),
        )

        # Only consider candidates with a positive BM25 score.
        selected = [
            candidate_id
            for candidate_id in ranked_candidates
            if best_candidate_score[candidate_id] > 0
        ][:CANDIDATE_POOL]

        documents = []

        for candidate_id in selected:
            documents.append({
                "candidate_id": candidate_id,
                "bm25_score": best_candidate_score[
                    candidate_id
                ],
                "chunks": parents[candidate_id],
            })

        print(
            "[RETRIEVE] Candidates with keyword matches:",
            len(documents),
        )

        return {"documents": documents}

    def verify(state: RAGState):
        print("\n[VERIFY] Checking all chunks per candidate")

        verified = []

        for document in state["documents"]:
            evidence = find_skill_evidence(
                document["chunks"],
                state["required_skills"],
            )

            status = classify_match(evidence)

            verified.append({
                **document,
                "evidence": evidence,
                "match_status": status,
            })

        full = [
            doc
            for doc in verified
            if doc["match_status"] == "FULL MATCH"
        ]

        partial = [
            doc
            for doc in verified
            if doc["match_status"] == "PARTIAL MATCH"
        ]

        print("[VERIFY] Full matches:", len(full))
        print("[VERIFY] Partial matches:", len(partial))

        # Show up to five candidates, prioritizing full matches.
        for doc in (full + partial)[:TOP_K]:
            print(
                f"  {doc['candidate_id']}: "
                f"{doc['match_status']}"
            )

            for skill, chunk_ids in doc["evidence"].items():
                print(
                    f"    {skill}: "
                    f"{chunk_ids if chunk_ids else 'NOT FOUND'}"
                )

        # Preserve full matches first.
        return {
            "documents": (full + partial)[:TOP_K]
        }

    def rewrite(state: RAGState):
        attempt = state["retry_count"] + 1

        print(f"\n[REWRITE] Attempt {attempt}")

        response = llm.invoke([
            (
                "system",
                "Rewrite a recruiter query for BM25 CV search. "
                "Preserve every mandatory technology exactly. "
                "Use plain language and useful synonyms. "
                "Do not use Boolean operators. "
                "Return only the rewritten query."
            ),
            (
                "human",
                f"Original query:\n{state['query']}\n\n"
                f"Previous search:\n"
                f"{state.get('rewritten_query') or state['query']}"
            ),
        ])

        rewritten = str(response.content).strip()

        if not rewritten:
            rewritten = state["query"]

        print("[REWRITE] New query:", rewritten)

        return {
            "rewritten_query": rewritten,
            "retry_count": attempt,
        }

    def generate(state: RAGState):
        print("\n[GENERATE] Preparing verified recommendations")

        full_matches = [
            doc
            for doc in state["documents"]
            if doc["match_status"] == "FULL MATCH"
        ]

        # Include the specific chunks supporting required skills.
        evidence_blocks = []

        for doc in full_matches:
            supporting_ids = set()

            for chunk_ids in doc["evidence"].values():
                supporting_ids.update(chunk_ids)

            supporting_chunks = [
                chunk
                for chunk in doc["chunks"]
                if chunk["chunk_id"] in supporting_ids
            ]

            chunk_text = "\n\n".join(
                f"Chunk ID: {chunk['chunk_id']}\n"
                f"Text: {chunk['text']}"
                for chunk in supporting_chunks
            )

            evidence_blocks.append(
                f"Candidate: {doc['candidate_id']}\n"
                f"Verified skill mentions: "
                f"{json.dumps(doc['evidence'])}\n"
                f"Supporting CV chunks:\n{chunk_text}"
            )

        evidence_text = "\n\n---\n\n".join(
            evidence_blocks
        )

        response = llm.invoke([
            (
                "system",
                "You are a careful recruiting research assistant. "
                "Use ONLY the supplied CV evidence. "
                "All candidates provided have explicit mentions "
                "of each required technology somewhere in their CV. "
                "This does NOT prove hands-on professional experience. "
                "Distinguish skills listed from project experience. "
                "Cite candidate IDs and supporting chunk IDs. "
                "Never invent names, qualifications, or experience. "
                "Explain uncertainty clearly."
            ),
            (
                "human",
                f"Recruiter query:\n{state['query']}\n\n"
                f"Mandatory technologies:\n"
                f"{state['required_skills']}\n\n"
                f"Verified evidence:\n{evidence_text}"
            ),
        ])

        return {
            "generation": str(response.content)
        }

    def fallback(state: RAGState):
        print("\n[FALLBACK] No full evidence match")

        return {
            "generation": (
                "No candidate was verified to have explicit "
                "CV mentions of every required technology: "
                f"{', '.join(state['required_skills'])}.\n"
                "This does not prove that no candidate has "
                "those skills; it means the indexed evidence "
                "did not establish a full match."
            )
        }

    def route(
        state: RAGState,
    ) -> Literal["generate", "rewrite", "fallback"]:

        full_matches = [
            doc
            for doc in state["documents"]
            if doc["match_status"] == "FULL MATCH"
        ]

        if full_matches:
            print("[ROUTER] -> GENERATE")
            return "generate"

        if state["retry_count"] < state["max_retries"]:
            print("[ROUTER] -> REWRITE")
            return "rewrite"

        print("[ROUTER] -> FALLBACK")
        return "fallback"

    workflow = StateGraph(RAGState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("verify", verify)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("fallback", fallback)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "verify")

    workflow.add_conditional_edges(
        "verify",
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

    query = (
        " ".join(sys.argv[1:])
        if len(sys.argv) > 1
        else "Find candidates with Python and Docker"
    )

    required_skills = extract_required_skills(query)

    print("=== LESSON 4E: PARENT CV VERIFICATION ===")
    print("Query:", query)
    print("Required skills:", required_skills)

    if not required_skills:
        print(
            "\nThis lesson supports explicit technology "
            "requirements from the public benchmark.\n"
            "Try: Find candidates with Python and Docker"
        )
        return

    chunks = load_chunks()

    print("Indexed chunks:", len(chunks))
    print(
        "Unique candidates:",
        len(collect_candidate_chunks(chunks)),
    )

    app = build_graph(chunks)

    initial_state = {
        "query": query,
        "rewritten_query": "",
        "required_skills": required_skills,
        "documents": [],
        "generation": "",
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
    print("Required skills:", result["required_skills"])
    print("Retries:", result["retry_count"])

    full_count = sum(
        doc["match_status"] == "FULL MATCH"
        for doc in result["documents"]
    )

    print("Full matches returned:", full_count)


if __name__ == "__main__":
    main()
