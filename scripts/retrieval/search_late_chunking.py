
"""Lesson 3: Compare retrieval using traditional and late chunk embeddings."""

import torch
import torch.nn.functional as F

from cv_intelligence.advanced_rag.late_chunking import (
    load_model,
    split_into_chunks,
    traditional_chunk_embeddings,
    late_chunk_embeddings,
    encode_token_ids,
    mean_pool,
)


def embed_query(query, tokenizer, model, device):
    """Embed a search query using the same BERT model."""

    token_ids = tokenizer.encode(
        query,
        add_special_tokens=False,
    )

    token_vectors = encode_token_ids(
        token_ids,
        tokenizer,
        model,
        device,
    )

    return mean_pool(token_vectors).cpu()


def rank_chunks(query_embedding, chunk_embeddings):
    """Rank chunks by cosine similarity."""

    scores = F.cosine_similarity(
        chunk_embeddings,
        query_embedding.unsqueeze(0),
        dim=1,
    )

    return torch.argsort(
        scores,
        descending=True,
    ).tolist(), scores.tolist()


def main():
    tokenizer, model, device = load_model()

    text = (
        "Sarah is a backend engineer at a fintech company. "
        "She develops secure payment systems and maintains "
        "production services. "
        "Her responsibilities include designing APIs, "
        "monitoring deployments, and improving reliability. "
        "She uses Python, PostgreSQL, and Docker. "
        "Her team deploys applications to cloud infrastructure."
    )

    query = "Which programming language does Sarah use?"
    chunk_size = 16

    chunks = split_into_chunks(
        text, tokenizer, chunk_size
    )

    traditional = traditional_chunk_embeddings(
        text, tokenizer, model, device, chunk_size
    )

    late = late_chunk_embeddings(
        text, tokenizer, model, device, chunk_size
    )

    query_embedding = embed_query(
        query, tokenizer, model, device
    )

    traditional_ranking, traditional_scores = rank_chunks(
        query_embedding, traditional
    )

    late_ranking, late_scores = rank_chunks(
        query_embedding, late
    )

    print("=== LATE CHUNKING RETRIEVAL DEMO ===")
    print("Query:", query)

    print("\n=== TRADITIONAL RANKING ===")
    for index in traditional_ranking:
        print(
            f"Chunk {index + 1} "
            f"(score={traditional_scores[index]:.4f}): "
            f"{tokenizer.decode(chunks[index])}"
        )

    print("\n=== LATE CHUNKING RANKING ===")
    for index in late_ranking:
        print(
            f"Chunk {index + 1} "
            f"(score={late_scores[index]:.4f}): "
            f"{tokenizer.decode(chunks[index])}"
        )


if __name__ == "__main__":
    main()
