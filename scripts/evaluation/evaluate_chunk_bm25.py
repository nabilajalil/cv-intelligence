
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from scripts.evaluation.evaluate_public_bm25 import (
    DATA_DIR,
    TOP_K,
    load_queries,
    load_qrels,
    ndcg_at_k,
)


CHUNKS_PATH = DATA_DIR / "chunks.jsonl"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def retrieve_candidates(
    query: str,
    chunks: list[dict],
    top_k: int = TOP_K,
) -> list[str]:
    """Rank chunks with BM25, then return unique candidate IDs."""

    if not chunks or top_k <= 0:
        return []

    tokenized_chunks = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_chunks)
    scores = bm25.get_scores(tokenize(query))

    ranked_indices = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True,
    )

    candidate_ids = []
    seen = set()

    for index in ranked_indices:
        candidate_id = chunks[index]["candidate_id"]

        if candidate_id in seen:
            continue

        seen.add(candidate_id)
        candidate_ids.append(candidate_id)

        if len(candidate_ids) == top_k:
            break

    return candidate_ids


def main():
    chunks = load_chunks()
    queries = load_queries()
    qrels = load_qrels()

    print("=== CHUNK-LEVEL BM25 EVALUATION ===")
    print("Chunks:", len(chunks))
    print("Candidates:", len({c["candidate_id"] for c in chunks}))
    print("Queries:", len(queries))
    print("Top-k:", TOP_K)

    metrics = {
        "precision": [],
        "recall": [],
        "ndcg": [],
    }

    for query in queries:
        query_id = query["query_id"]
        candidate_ids = retrieve_candidates(
            query["text"],
            chunks,
        )

        labels = qrels[query_id]
        grades = [
            labels.get(candidate_id, 0)
            for candidate_id in candidate_ids
        ]

        full_matches = sum(
            grade == 2
            for grade in labels.values()
        )
        retrieved_full = sum(
            grade == 2
            for grade in grades
        )

        precision = retrieved_full / TOP_K
        recall = (
            retrieved_full / full_matches
            if full_matches else None
        )
        ndcg = ndcg_at_k(
            grades,
            list(labels.values()),
            TOP_K,
        )

        metrics["precision"].append(precision)

        if recall is not None:
            metrics["recall"].append(recall)

        if ndcg is not None:
            metrics["ndcg"].append(ndcg)

        print(f"\n{query_id}: {query['text']}")
        print("Retrieved IDs:", candidate_ids)
        print("Relevance grades:", grades)
        print(f"Precision@5: {precision:.2f}")
        print(
            f"Recall@5: {recall:.2f}"
            if recall is not None
            else "Recall@5: N/A"
        )
        print(
            f"nDCG@5: {ndcg:.2f}"
            if ndcg is not None
            else "nDCG@5: N/A"
        )

    print("\n=== MEAN METRICS ===")

    for name, values in metrics.items():
        if values:
            mean = sum(values) / len(values)
            print(f"Mean {name}@5: {mean:.3f}")


if __name__ == "__main__":
    main()

