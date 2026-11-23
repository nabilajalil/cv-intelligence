
"""Stage 3E: Evaluate traditional vs native Jina Late Chunking."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from scripts.demo_jina_late_chunking import embed
from scripts.evaluation.evaluate_public_bm25 import ndcg_at_k


DATA_DIR = Path("data/public_benchmark")
CACHE_DIR = DATA_DIR / "jina_embedding_cache"
TOP_K = 5


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_qrels(path):
    qrels = defaultdict(dict)

    with path.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            qrels[row["query_id"]][
                row["candidate_id"]
            ] = int(row["relevance"])

    return qrels


def cached_embeddings(name, texts, task, late_chunking):
    """Reuse saved embeddings if available."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.npz"

    # A content fingerprint prevents stale-cache comparisons.
    import hashlib

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "model": "jina-embeddings-v3",
                "texts": texts,
                "task": task,
                "late_chunking": late_chunking,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if str(saved["fingerprint"]) == fingerprint:
                print(f"Cached: {name}")
                return saved["vectors"].copy()

    print(f"Embedding: {name} ({len(texts)} texts)")

    vectors = embed(
        texts,
        task=task,
        late_chunking=late_chunking,
    )

    np.savez_compressed(
        path,
        vectors=vectors,
        fingerprint=fingerprint,
    )

    return vectors


def build_vectors(chunks, late_chunking):
    """Embed each CV independently in original chunk order."""

    by_candidate = defaultdict(list)

    for chunk in chunks:
        by_candidate[chunk["candidate_id"]].append(chunk)

    vectors_by_id = {}

    mode = "late" if late_chunking else "traditional"

    for candidate_id in sorted(by_candidate):
        candidate_chunks = sorted(
            by_candidate[candidate_id],
            key=lambda item: item["chunk_index"],
        )

        texts = [
            chunk["text"]
            for chunk in candidate_chunks
        ]

        # Each request contains chunks from ONE CV only.
        vectors = cached_embeddings(
            name=f"{mode}_{candidate_id}",
            texts=texts,
            task="retrieval.passage",
            late_chunking=late_chunking,
        )

        if len(vectors) != len(candidate_chunks):
            raise ValueError(
                f"Unexpected vector count for {candidate_id}"
            )

        for chunk, vector in zip(candidate_chunks, vectors):
            vectors_by_id[chunk["chunk_id"]] = vector

    return np.stack([
        vectors_by_id[chunk["chunk_id"]]
        for chunk in chunks
    ])


def normalize(vectors):
    norms = np.linalg.norm(
        vectors, axis=1, keepdims=True
    )
    return vectors / np.maximum(norms, 1e-12)


def rank_candidates(query_vector, vectors, chunks):
    """Use each candidate's highest-scoring chunk."""

    scores = normalize(vectors) @ normalize(
        query_vector.reshape(1, -1)
    )[0]

    best_scores = {}

    for chunk, score in zip(chunks, scores):
        candidate_id = chunk["candidate_id"]

        if (
            candidate_id not in best_scores
            or score > best_scores[candidate_id]
        ):
            best_scores[candidate_id] = float(score)

    return sorted(
        best_scores,
        key=lambda candidate_id: (
            -best_scores[candidate_id],
            candidate_id,
        ),
    )[:TOP_K]


def evaluate(name, vectors, query_vectors, chunks, queries, qrels):
    candidate_ids = {
        chunk["candidate_id"]
        for chunk in chunks
    }

    totals = {
        "precision": 0.0,
        "recall": 0.0,
        "ndcg": 0.0,
    }

    print(f"\n=== {name} ===")

    for query, query_vector in zip(queries, query_vectors):
        query_id = query["query_id"]

        ranked = rank_candidates(
            query_vector, vectors, chunks
        )

        judgments = {
            candidate_id: qrels[query_id].get(
                candidate_id, 0
            )
            for candidate_id in candidate_ids
        }

        grades = [
            judgments.get(candidate_id, 0)
            for candidate_id in ranked
        ]

        all_grades = list(judgments.values())

        # Grade 2 = both requested skills.
        relevant_retrieved = sum(
            grade == 2 for grade in grades
        )

        total_relevant = sum(
            grade == 2 for grade in all_grades
        )

        precision = relevant_retrieved / TOP_K

        recall = (
            relevant_retrieved / total_relevant
            if total_relevant
            else 0.0
        )

        ndcg = ndcg_at_k(
            grades, all_grades, TOP_K
        )

        totals["precision"] += precision
        totals["recall"] += recall
        totals["ndcg"] += ndcg

        print(f"\n{query_id}: {query['text']}")
        print("Retrieved:", ranked)
        print("Grades:", grades)
        print(
            f"P@5={precision:.3f} "
            f"R@5={recall:.3f} "
            f"nDCG@5={ndcg:.3f}"
        )

    return {
        metric: value / len(queries)
        for metric, value in totals.items()
    }


def main():
    load_dotenv()

    chunks = load_jsonl(DATA_DIR / "chunks.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")
    qrels = load_qrels(DATA_DIR / "qrels.tsv")

    candidate_ids = {
        chunk["candidate_id"] for chunk in chunks
    }

    print("=== STAGE 3E: JINA BENCHMARK ===")
    print("Candidates:", len(candidate_ids))
    print("Chunks:", len(chunks))
    print("Queries:", len(queries))

    # Embed recruiter queries once for both approaches.
    query_vectors = cached_embeddings(
        name="queries",
        texts=[query["text"] for query in queries],
        task="retrieval.query",
        late_chunking=False,
    )

    print("\n--- Traditional indexing ---")
    traditional_vectors = build_vectors(
        chunks, late_chunking=False
    )

    print("\n--- Late Chunking indexing ---")
    late_vectors = build_vectors(
        chunks, late_chunking=True
    )

    traditional = evaluate(
        "TRADITIONAL JINA",
        traditional_vectors,
        query_vectors,
        chunks,
        queries,
        qrels,
    )

    late = evaluate(
        "NATIVE LATE CHUNKING",
        late_vectors,
        query_vectors,
        chunks,
        queries,
        qrels,
    )

    print("\n=== FINAL COMPARISON ===")
    print("Metric          Traditional    Late")
    print("-----------------------------------")

    for metric in ("precision", "recall", "ndcg"):
        print(
            f"{metric:<15}"
            f"{traditional[metric]:.3f}          "
            f"{late[metric]:.3f}"
        )

    print("\nCache directory:", CACHE_DIR)


if __name__ == "__main__":
    main()
