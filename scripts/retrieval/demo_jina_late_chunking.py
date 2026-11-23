
"""Stage 3D: Native Late Chunking with Jina Embeddings v3."""

import os

import numpy as np
import requests
from dotenv import load_dotenv

from scripts.evaluation.evaluate_chunk_bm25 import load_chunks


API_URL = "https://api.jina.ai/v1/embeddings"
MODEL = "jina-embeddings-v3"


def embed(texts, task, late_chunking=False):
    """Call Jina and return vectors in input order."""

    api_key = os.getenv("JINA_API_KEY")

    if not api_key:
        raise ValueError(
            "JINA_API_KEY is missing from your .env file."
        )

    payload = {
        "model": MODEL,
        "task": task,
        "input": texts,
        "late_chunking": late_chunking,
        "embedding_type": "float",
        "truncate": False,
    }

    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    if not response.ok:
        raise RuntimeError(
            f"Jina API error {response.status_code}: "
            f"{response.text[:500]}"
        )

    data = response.json()["data"]

    # Use response indices to preserve chunk order.
    data = sorted(data, key=lambda item: item["index"])

    if len(data) != len(texts):
        raise RuntimeError(
            "Jina returned an unexpected number of vectors."
        )

    return np.asarray(
        [item["embedding"] for item in data],
        dtype=np.float32,
    )


def cosine_scores(vectors, query_vector):
    """Calculate cosine similarity for every chunk."""

    vector_norms = np.linalg.norm(
        vectors, axis=1, keepdims=True
    )

    normalized_vectors = vectors / np.maximum(
        vector_norms, 1e-12
    )

    normalized_query = query_vector / max(
        np.linalg.norm(query_vector), 1e-12
    )

    return normalized_vectors @ normalized_query


def show_ranking(name, chunks, scores):
    print(f"\n=== {name} ===")

    ranking = np.argsort(scores)[::-1]

    for index in ranking:
        print(
            f"\nChunk {index + 1} "
            f"(score={scores[index]:.4f})"
        )
        print(chunks[index]["text"][:220])


def main():
    load_dotenv()

    # Reuse the existing 224-chunk benchmark.
    all_chunks = load_chunks()

    # Start with one CV to keep the API experiment small.
    candidate_id = "r00001"

    chunks = sorted(
        [
            chunk for chunk in all_chunks
            if chunk["candidate_id"] == candidate_id
        ],
        key=lambda chunk: chunk["chunk_index"],
    )

    if not chunks:
        raise ValueError(
            f"No chunks found for {candidate_id}."
        )

    texts = [chunk["text"] for chunk in chunks]

    query = (
        "Which candidate has experience with "
        "Docker and GitLab?"
    )

    print("=== JINA NATIVE LATE CHUNKING DEMO ===")
    print("Candidate:", candidate_id)
    print("Chunks:", len(chunks))
    print("Query:", query)

    # Traditional: chunks are embedded independently.
    traditional_vectors = embed(
        texts,
        task="retrieval.passage",
        late_chunking=False,
    )

    # Native late chunking: all chunks from this CV
    # are processed together in document order.
    late_vectors = embed(
        texts,
        task="retrieval.passage",
        late_chunking=True,
    )

    # Embed the query separately.
    query_vector = embed(
        [query],
        task="retrieval.query",
        late_chunking=False,
    )[0]

    traditional_scores = cosine_scores(
        traditional_vectors,
        query_vector,
    )

    late_scores = cosine_scores(
        late_vectors,
        query_vector,
    )

    print("\nEmbedding dimensions:")
    print("Traditional:", traditional_vectors.shape)
    print("Late:", late_vectors.shape)

    show_ranking(
        "TRADITIONAL CHUNKING",
        chunks,
        traditional_scores,
    )

    show_ranking(
        "NATIVE LATE CHUNKING",
        chunks,
        late_scores,
    )

    print("\n=== VECTOR DIFFERENCES ===")

    for index, (early, late) in enumerate(
        zip(traditional_vectors, late_vectors),
        start=1,
    ):
        similarity = cosine_scores(
            early.reshape(1, -1),
            late,
        )[0]

        print(
            f"Chunk {index}: "
            f"early-vs-late cosine={similarity:.4f}"
        )


if __name__ == "__main__":
    main()
