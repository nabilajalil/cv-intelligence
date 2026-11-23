
"""Lesson 4B: Hybrid Agentic RAG for CV Intelligence.

Retrieval: BM25 + cached Jina native Late Chunking + RRF
Agent: retrieve -> grade -> generate / rewrite / fallback

Requires the cached embeddings from Stage 3E.
"""

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from rank_bm25 import BM25Okapi

from scripts.retrieval.demo_jina_late_chunking import embed


DATA_DIR = Path("data/public_benchmark")
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CACHE_DIR = DATA_DIR / "jina_embedding_cache"

MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "jina-embeddings-v3"

TOP_K = 5
RETRIEVAL_POOL = 20
RRF_K = 60
MAX_RETRIES = 2


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
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def fingerprint(texts, task, late_chunking):
    payload = {
        "model": EMBEDDING_MODEL,
        "texts": texts,
        "task": task,
        "late_chunking": late_chunking,
    }

    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    return vectors / np.maximum(norms, 1e-12)


def load_late_vectors(chunks):
    """Load and validate all Stage 3E cached CV embeddings."""

    by_candidate = defaultdict(list)

    for chunk in chunks:
        by_candidate[chunk["candidate_id"]].append(chunk)

    vectors_by_chunk = {}

    for candidate_id in sorted(by_candidate):
        candidate_chunks = sorted(
            by_candidate[candidate_id],
            key=lambda item: item["chunk_index"],
        )

        texts = [
            chunk["text"]
            for chunk in candidate_chunks
        ]

        path = CACHE_DIR / f"late_{candidate_id}.npz"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing cached vectors: {path}\n"
                "Run Stage 3E first:\n"
                "python -m scripts.evaluation.evaluate_jina_late_chunking"
            )

        with np.load(path, allow_pickle=False) as saved:
            vectors = saved["vectors"].copy()
            saved_fingerprint = str(saved["fingerprint"])

        expected = fingerprint(
            texts,
            task="retrieval.passage",
            late_chunking=True,
        )

        if saved_fingerprint != expected:
            raise ValueError(
                f"Cache does not match current chunks: {path}"
            )

        if len(vectors) != len(candidate_chunks):
            raise ValueError(
                f"Wrong vector count for {candidate_id}"
            )

        for chunk, vector in zip(candidate_chunks, vectors):
            vectors_by_chunk[chunk["chunk_id"]] = vector

    ordered_vectors = np.stack([
        vectors_by_chunk[chunk["chunk_id"]]
        for chunk in chunks
    ])

    return normalize(ordered_vectors)


def embed_query(query):
    """Cache Jina query vectors by query text."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key = fingerprint(
        [query],
        task="retrieval.query",
        late_chunking=False,
    )

    path = CACHE_DIR / f"hybrid_query_{key}.npz"

    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            vector = saved["vector"].copy()

        print("[JINA] Reusing cached query embedding")
        return vector

    print("[JINA] Embedding new recruiter query")

    vector = embed(
        [query],
        task="retrieval.query",
        late_chunking=False,
    )[0]

    np.savez_compressed(path, vector=vector)

    return vector


def candidate_ranking(scores, chunks, limit):
    """Rank unique candidates using their best chunk."""

    best = {}

    for index, (score, chunk) in enumerate(zip(scores, chunks)):
        candidate_id = chunk["candidate_id"]

        if (
            candidate_id not in best
            or score > best[candidate_id]["score"]
        ):
            best[candidate_id] = {
                "score": float(score),
                "chunk_index": index,
            }

    ranked = sorted(
        best,
        key=lambda candidate_id: (
            -best[candidate_id]["score"],
            candidate_id,
        ),
    )

    return ranked[:limit], best


def reciprocal_rank_fusion(bm25_ranked, jina_ranked):
    """Fuse two candidate rankings using RRF."""

    fused = defaultdict(float)

    for ranking in (bm25_ranked, jina_ranked):
        for rank, candidate_id in enumerate(ranking, start=1):
            fused[candidate_id] += 1.0 / (RRF_K + rank)

    return sorted(
        fused,
        key=lambda candidate_id: (
            -fused[candidate_id],
            candidate_id,
        ),
    ), fused


def build_graph(chunks, late_vectors):
    llm = ChatOpenAI(
        model=MODEL,
        temperature=0,
    )

    bm25 = BM25Okapi([
        tokenize(chunk["text"])
        for chunk in chunks
    ])

    def retrieve(state: RAGState):
        query = (
            state.get("rewritten_query")
            or state["query"]
        )

        print(f"\n[RETRIEVE] {query}")

        # Branch 1: exact keyword retrieval.
        bm25_scores = bm25.get_scores(tokenize(query))

        bm25_ranked, bm25_best = candidate_ranking(
            bm25_scores,
            chunks,
            RETRIEVAL_POOL,
        )

        # Branch 2: semantic retrieval.
        query_vector = normalize(
            embed_query(query).reshape(1, -1)
        )[0]

        jina_scores = late_vectors @ query_vector

        jina_ranked, jina_best = candidate_ranking(
            jina_scores,
            chunks,
            RETRIEVAL_POOL,
        )

        # Combine candidate-level rankings.
        fused_ranked, fused_scores = reciprocal_rank_fusion(
            bm25_ranked,
            jina_ranked,
        )

        print("[BM25] Top 5:", bm25_ranked[:5])
        print("[JINA] Top 5:", jina_ranked[:5])
        print("[RRF]  Top 5:", fused_ranked[:5])

        documents = []

        for candidate_id in fused_ranked[:TOP_K]:
            # Select a chunk from the best BM25 or Jina
            # match. Prefer the BM25 chunk when its
            # score is positive; otherwise use Jina.
            bm25_match = bm25_best.get(candidate_id)
            jina_match = jina_best.get(candidate_id)

            if (
                bm25_match is not None
                and bm25_match["score"] > 0
            ):
                index = bm25_match["chunk_index"]
            elif jina_match is not None:
                index = jina_match["chunk_index"]
            else:
                continue

            chunk = chunks[index]

            documents.append({
                "candidate_id": candidate_id,
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "rrf_score": fused_scores[candidate_id],
            })

        print(
            f"[RETRIEVE] Fused {len(documents)} candidates"
        )

        return {"documents": documents}

    def grade(state: RAGState):
        print("\n[GRADE] Evaluating CV evidence")

        relevant = []
        scores = []

        for document in state["documents"]:
            response = llm.invoke([
                (
                    "system",
                    "You are a strict CV evidence grader. "
                    "Evaluate the candidate evidence against "
                    "the ORIGINAL recruiter query. "
                    "Return only one number: "
                    "0.0, 0.3, 0.7, or 1.0. "
                    "1.0 means the evidence clearly satisfies "
                    "all mandatory requirements. "
                    "0.7 means strong relevant evidence with "
                    "some uncertainty. "
                    "0.3 means partial or weak evidence. "
                    "0.0 means unrelated. "
                    "Do not invent skills or experience."
                ),
                (
                    "human",
                    f"Recruiter query:\n{state['query']}\n\n"
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
                f"  {document['candidate_id']}: {score:.2f}"
            )

            if score >= 0.7:
                relevant.append({
                    **document,
                    "relevance_score": score,
                })

        average = (
            sum(scores) / len(scores)
            if scores else 0.0
        )

        print(
            f"[GRADE] Kept {len(relevant)}/"
            f"{len(state['documents'])}"
        )

        return {
            "documents": relevant,
            "relevance_score": average,
        }

    def rewrite(state: RAGState):
        attempt = state["retry_count"] + 1

        print(f"\n[REWRITE] Attempt {attempt}")

        response = llm.invoke([
            (
                "system",
                "Rewrite the recruiter query for hybrid "
                "keyword and semantic CV search. "
                "Preserve all mandatory requirements. "
                "Use natural language with useful synonyms. "
                "Do NOT use Boolean operators, parentheses, "
                "quotation marks, AND, OR, or NOT. "
                "Return only the new search query."
            ),
            (
                "human",
                f"Original recruiter query:\n"
                f"{state['query']}\n\n"
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
        print("\n[GENERATE] Building grounded answer")

        evidence = "\n\n".join(
            f"Candidate: {doc['candidate_id']}\n"
            f"Chunk: {doc['chunk_id']}\n"
            f"Evidence:\n{doc['text']}"
            for doc in state["documents"]
        )

        response = llm.invoke([
            (
                "system",
                "You are a recruiting research assistant. "
                "Answer ONLY from supplied CV evidence. "
                "Cite candidate IDs and chunk IDs. "
                "Distinguish explicitly listed skills from "
                "hands-on project experience. "
                "Never describe a skill as demonstrated "
                "project experience unless the evidence "
                "actually connects it to specific work. "
                "Do not invent qualifications. "
                "A chunk is partial evidence, not a full CV."
            ),
            (
                "human",
                f"Recruiter query:\n{state['query']}\n\n"
                f"CV evidence:\n{evidence}"
            ),
        ])

        return {
            "generation": str(response.content)
        }

    def fallback(state: RAGState):
        print("\n[FALLBACK] No sufficient CV evidence")

        return {
            "generation": (
                "I couldn't find sufficiently relevant "
                "CV evidence for this recruiter request: "
                f"{state['query']}\n"
                "Consider adjusting the requirements or "
                "expanding the candidate collection."
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
    import sys

    load_dotenv()

    chunks = load_chunks()

    print("=== LESSON 4B: HYBRID AGENTIC RAG ===")
    print("Chunks:", len(chunks))

    print("\nLoading cached Jina Late Chunking vectors...")
    late_vectors = load_late_vectors(chunks)

    print("Vector shape:", late_vectors.shape)

    app = build_graph(chunks, late_vectors)

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
        "Final search:",
        result["rewritten_query"] or result["query"],
    )
    print("Retries:", result["retry_count"])
    print("Relevant candidates:", len(result["documents"]))


if __name__ == "__main__":
    main()
