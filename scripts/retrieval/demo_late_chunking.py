
"""Compare traditional and late chunk embeddings."""

from cv_intelligence.advanced_rag.late_chunking import (
    load_model,
    split_into_chunks,
    traditional_chunk_embeddings,
    late_chunk_embeddings,
    cosine_similarity,
)


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

    chunk_size = 16

    chunks = split_into_chunks(
        text, tokenizer, chunk_size
    )

    print("=== LATE CHUNKING DEMO ===")
    print("Device:", device)
    print("Model:", model.config.model_type)
    print("Number of chunks:", len(chunks))

    print("\n=== CHUNKS ===")

    for index, chunk_ids in enumerate(chunks):
        print(
            f"Chunk {index + 1}:",
            tokenizer.decode(chunk_ids),
        )

    traditional = traditional_chunk_embeddings(
        text, tokenizer, model, device, chunk_size
    )

    late = late_chunk_embeddings(
        text, tokenizer, model, device, chunk_size
    )

    print("\n=== EMBEDDING SHAPES ===")
    print("Traditional:", tuple(traditional.shape))
    print("Late:", tuple(late.shape))

    print("\n=== COSINE SIMILARITY ===")

    for index in range(len(chunks)):
        similarity = cosine_similarity(
            traditional[index],
            late[index],
        )

        print(
            f"Chunk {index + 1}: "
            f"{similarity:.4f}"
        )


if __name__ == "__main__":
    main()
