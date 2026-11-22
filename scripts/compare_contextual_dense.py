
"""Stage 3B: Compare original vs contextual dense retrieval."""

import csv
import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from scripts.evaluate_public_bm25 import ndcg_at_k


DATA_DIR = Path("data/public_benchmark")
CONTEXTUAL_PATH = DATA_DIR / "llm_contextual_chunks.jsonl"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
QRELS_PATH = DATA_DIR / "qrels.tsv"
CACHE_PATH = DATA_DIR / "dense_embedding_cache.json"

MODEL = "text-embedding-3-small"
TOP_K = 5


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def load_qrels(path):
    qrels = {}

    with path.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            query_id = row["query_id"]
            candidate_id = row["candidate_id"]
            relevance = int(row["relevance"])

            qrels.setdefault(query_id, {})[
                candidate_id
            ] = relevance

    return qrels


def load_cache():
    if CACHE_PATH.exists():
        with CACHE_PATH.open(
            "r", encoding="utf-8"
        ) as file:
            return json.load(file)

    return {}


def save_cache(cache):
    with CACHE_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(cache, file)


def embed_texts(texts, keys, client, cache):
    """Embed only texts not already cached."""

    missing_keys = [
        key for key in keys if key not in cache
    ]

    if missing_keys:
        text_by_key = dict(zip(keys, texts))

        print(
            f"Embedding {len(missing_keys)} "
            "new texts..."
        )

        # Batch requests to reduce API overhead.
        batch_size = 50

        for start in range(
            0, len(missing_keys), batch_size
        ):
            batch_keys = missing_keys[
                start:start + batch_size
            ]

            batch_texts = [
                text_by_key[key]
                for key in batch_keys
            ]

            response = client.embeddings.create(
                model=MODEL,
                input=batch_texts,
            )

            for item in response.data:
                key = batch_keys[item.index]
                cache[key] = item.embedding

            save_cache(cache)

    return np.asarray(
        [cache[key] for key in keys],
        dtype=np.float32,
    )


def normalize(vectors):
    norms = np.linalg.norm(
        vectors, axis=1, keepdims=True
    )

    return vectors / np.maximum(norms, 1e-12)


def retrieve_candidates(
    query_vector,
    chunk_vectors,
    chunks,
    top_k=TOP_K,
):
    """Rank candidates by their best-matching chunk."""

    scores = normalize(chunk_vectors) @ normalize(
        query_vector.reshape(1, -1)
    )[0]

    best_by_candidate = {}

    for chunk, score in zip(chunks, scores):
        candidate_id = chunk["candidate_id"]

        if (
            candidate_id not in best_by_candidate
            or score > best_by_candidate[candidate_id]
        ):
            best_by_candidate[candidate_id] = float(
                score
            )

    ranked = sorted(
        best_by_candidate,
        key=best_by_candidate.get,
        reverse=True,
    )

    return ranked[:top_k]


def evaluate(
    name,
    chunk_vectors,
    query_vectors,
    chunks,
    queries,
    qrels,
    candidate_ids,
):
    totals = {
        "precision": 0.0,
        "recall": 0.0,
        "ndcg": 0.0,
    }

    print(f"\n=== {name} ===")

    for query, query_vector in zip(
        queries, query_vectors
    ):
        query_id = query["query_id"]

        retrieved = retrieve_candidates(
            query_vector,
            chunk_vectors,
            chunks,
        )

        judgments = {
            candidate_id: qrels[query_id].get(
                candidate_id, 0
            )
            for candidate_id in candidate_ids
        }

        retrieved_grades = [
            judgments.get(candidate_id, 0)
            for candidate_id in retrieved
        ]

        all_grades = list(judgments.values())

        # Our benchmark considers grade 2 fully relevant.
        relevant_retrieved = sum(
            grade == 2
            for grade in retrieved_grades
        )

        total_relevant = sum(
            grade == 2
            for grade in all_grades
        )

        precision = relevant_retrieved / TOP_K

        recall = (
            relevant_retrieved / total_relevant
            if total_relevant
            else 0.0
        )

        ndcg = ndcg_at_k(
            retrieved_grades,
            all_grades,
            TOP_K,
        )

        totals["precision"] += precision
        totals["recall"] += recall
        totals["ndcg"] += ndcg

        print(
            f"{query_id}: "
            f"P@5={precision:.3f} "
            f"R@5={recall:.3f} "
            f"nDCG@5={ndcg:.3f}"
        )
        print("  Retrieved:", retrieved)
        print("  Grades:", retrieved_grades)

    count = len(queries)

    return {
        metric: value / count
        for metric, value in totals.items()
    }


def main():
    load_dotenv()
    client = OpenAI()

    chunks = load_jsonl(CONTEXTUAL_PATH)
    queries = load_jsonl(QUERIES_PATH)
    qrels = load_qrels(QRELS_PATH)

    candidate_ids = sorted({
        chunk["candidate_id"]
        for chunk in chunks
    })

    print("=== STAGE 3B: DENSE RETRIEVAL ===")
    print("Chunks:", len(chunks))
    print("Candidates:", len(candidate_ids))
    print("Queries:", len(queries))

    if len(candidate_ids) < TOP_K:
        raise ValueError(
            "Not enough candidates for top-5 retrieval."
        )

    if not all(
        chunk.get("original_text")
        for chunk in chunks
    ):
        raise ValueError(
            "Missing original_text in contextual chunks."
        )

    original_texts = [
        chunk["original_text"]
        for chunk in chunks
    ]

    contextual_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    query_texts = [
        query["text"]
        for query in queries
    ]

    chunk_keys = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    cache = load_cache()

    original_vectors = embed_texts(
        original_texts,
        [
            f"{MODEL}:original:{key}"
            for key in chunk_keys
        ],
        client,
        cache,
    )

    contextual_vectors = embed_texts(
        contextual_texts,
        [
            f"{MODEL}:contextual:{key}"
            for key in chunk_keys
        ],
        client,
        cache,
    )

    query_vectors = embed_texts(
        query_texts,
        [
            f"{MODEL}:query:{query['query_id']}"
            for query in queries
        ],
        client,
        cache,
    )

    original = evaluate(
        "ORIGINAL DENSE RETRIEVAL",
        original_vectors,
        query_vectors,
        chunks,
        queries,
        qrels,
        candidate_ids,
    )

    contextual = evaluate(
        "CONTEXTUAL DENSE RETRIEVAL",
        contextual_vectors,
        query_vectors,
        chunks,
        queries,
        qrels,
        candidate_ids,
    )

    print("\n=== FINAL COMPARISON ===")
    print("Metric          Original     Contextual")
    print("---------------------------------------")

    for metric in ("precision", "recall", "ndcg"):
        print(
            f"{metric:<15}"
            f"{original[metric]:.3f}        "
            f"{contextual[metric]:.3f}"
        )

    print("\nEmbeddings cached at:", CACHE_PATH)


if __name__ == "__main__":
    main()
