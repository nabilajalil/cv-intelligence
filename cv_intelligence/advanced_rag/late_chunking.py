
"""Lesson 3: Traditional chunking vs Late Chunking."""

import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "bert-base-uncased"


def load_model():
    """Load BERT and select CPU or GPU."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    return tokenizer, model, device


def split_into_chunks(text, tokenizer, chunk_size=32):
    """Split text into chunks of token IDs."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    return [
        token_ids[i:i + chunk_size]
        for i in range(0, len(token_ids), chunk_size)
    ]


def mean_pool(token_vectors):
    """Average token vectors into one chunk embedding."""

    return token_vectors.mean(dim=0)


def encode_token_ids(token_ids, tokenizer, model, device):
    """Run BERT on token IDs with CLS and SEP tokens."""

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if cls_id is None or sep_id is None:
        raise ValueError("This demo requires a BERT tokenizer.")

    input_ids = [cls_id] + token_ids + [sep_id]

    inputs = {
        "input_ids": torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.ones(
            (1, len(input_ids)),
            dtype=torch.long,
            device=device,
        ),
    }

    with torch.inference_mode():
        outputs = model(**inputs)

    # Remove CLS and SEP representations.
    return outputs.last_hidden_state[0, 1:-1]


def traditional_chunk_embeddings(
    text,
    tokenizer,
    model,
    device,
    chunk_size=32,
):
    """Split first, then encode each chunk independently."""

    chunks = split_into_chunks(
        text, tokenizer, chunk_size
    )

    if not chunks:
        raise ValueError("Cannot embed empty text.")

    embeddings = []

    for chunk_ids in chunks:
        token_vectors = encode_token_ids(
            chunk_ids,
            tokenizer,
            model,
            device,
        )

        embedding = mean_pool(token_vectors)
        embeddings.append(embedding.cpu())

    return torch.stack(embeddings)


def late_chunk_embeddings(
    text,
    tokenizer,
    model,
    device,
    chunk_size=32,
):
    """Encode the full text, then pool by chunk boundaries."""

    chunks = split_into_chunks(
        text, tokenizer, chunk_size
    )

    if not chunks:
        raise ValueError("Cannot embed empty text.")

    token_ids = [
        token_id
        for chunk in chunks
        for token_id in chunk
    ]

    max_content_tokens = (
        model.config.max_position_embeddings - 2
    )

    if len(token_ids) > max_content_tokens:
        raise ValueError(
            f"Text has {len(token_ids)} tokens, "
            f"but this model supports at most "
            f"{max_content_tokens} content tokens. "
            "Use a shorter document for this demo."
        )

    # Encode all tokens together so they can share context.
    token_vectors = encode_token_ids(
        token_ids,
        tokenizer,
        model,
        device,
    )

    embeddings = []
    start = 0

    for chunk_ids in chunks:
        end = start + len(chunk_ids)

        embedding = mean_pool(
            token_vectors[start:end]
        )

        embeddings.append(embedding.cpu())
        start = end

    return torch.stack(embeddings)


def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""

    return F.cosine_similarity(
        a.unsqueeze(0),
        b.unsqueeze(0),
    ).item()
