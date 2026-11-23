
"""Lesson 6B.1 — Multimodal-derived CV retrieval.

Builds a local BM25 index from the structured profiles
produced by text and vision ingestion.

Keeps:
- Candidate identity
- Extraction method
- Source PDF
- Searchable profile content

No OpenAI API calls are required.

Run:
    python -m scripts.multimodal.demo_multimodal_retrieval

Optional:
    python -m scripts.multimodal.demo_multimodal_retrieval \
        --query "Python Docker engineer"
"""

import argparse
import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROFILE_DIR = Path(
    "data/multimodal/hybrid_profiles"
)

OUTPUT_DIR = Path(
    "data/multimodal/retrieval"
)

DEFAULT_QUERY = "Python Docker engineer"

TOP_K = 5


def tokenize(text):
    """Simple BM25 tokenization."""

    return re.findall(
        r"[a-z0-9]+",
        text.casefold(),
    )


def load_profiles():
    """Load both hybrid ingestion results."""

    paths = sorted(
        PROFILE_DIR.glob("*_profile.json")
    )

    if not paths:
        raise FileNotFoundError(
            f"No hybrid profiles found in {PROFILE_DIR}"
        )

    records = []

    for path in paths:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        profile = data.get("profile", {})

        candidate_id = profile.get(
            "candidate_id"
        )

        if not candidate_id:
            raise ValueError(
                f"Missing candidate ID in {path}"
            )

        records.append({
            "record_id": path.stem,
            "candidate_id": candidate_id,
            "extraction_method": data.get(
                "extraction_method"
            ),
            "source_file": data.get(
                "source_file"
            ),
            "profile": profile,
        })

    return records


def profile_to_text(profile):
    """Convert a structured CV into searchable text."""

    sections = []

    candidate_id = profile.get(
        "candidate_id"
    )

    role = profile.get("role")

    if candidate_id:
        sections.append(
            f"Candidate: {candidate_id}"
        )

    if role:
        sections.append(
            f"Role: {role}"
        )

    for field in [
        "skills",
        "experience",
        "projects",
        "education",
    ]:
        values = profile.get(field, [])

        if not isinstance(values, list):
            continue

        if values:
            sections.append(
                f"{field.title()}: "
                + " | ".join(values)
            )

    return "\n".join(sections)


def build_index(records):
    """Create a BM25 index over profile text."""

    documents = [
        profile_to_text(record["profile"])
        for record in records
    ]

    tokenized_documents = [
        tokenize(document)
        for document in documents
    ]

    if not any(tokenized_documents):
        raise ValueError(
            "No searchable profile text found."
        )

    bm25 = BM25Okapi(
        tokenized_documents
    )

    return bm25, documents


def search(query, records, bm25, documents):
    """Rank source records using BM25."""

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(records)),
        key=lambda index: (
            -float(scores[index]),
            records[index]["record_id"],
        ),
    )

    results = []

    for index in ranked_indices[:TOP_K]:
        record = records[index]

        document_tokens = set(
            tokenize(documents[index])
        )

        matched_terms = sorted(
            set(query_tokens)
            & document_tokens
        )

        results.append({
            "record_id": record["record_id"],
            "candidate_id": record["candidate_id"],
            "extraction_method": record[
                "extraction_method"
            ],
            "source_file": record["source_file"],
            "bm25_score": float(
                scores[index]
            ),
            "matched_terms": matched_terms,
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Search text and vision-derived CV profiles"
        )
    )

    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Candidate search query",
    )

    args = parser.parse_args()

    print(
        "=== LESSON 6B.1: "
        "MULTIMODAL CV RETRIEVAL ==="
    )

    records = load_profiles()

    bm25, documents = build_index(
        records
    )

    unique_candidates = {
        record["candidate_id"]
        for record in records
    }

    print("Source records:", len(records))
    print(
        "Unique candidates:",
        len(unique_candidates),
    )

    print("Query:", args.query)

    results = search(
        args.query,
        records,
        bm25,
        documents,
    )

    print("\nRETRIEVAL RESULTS")

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\nRank {rank}"
        )
        print(
            "Candidate:",
            result["candidate_id"],
        )
        print(
            "Extraction method:",
            result["extraction_method"],
        )
        print(
            "BM25 score:",
            f"{result['bm25_score']:.4f}",
        )
        print(
            "Matched terms:",
            result["matched_terms"],
        )
        print(
            "Source:",
            result["source_file"],
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "multimodal_search_results.json"
    )

    output_path.write_text(
        json.dumps(
            {
                "query": args.query,
                "source_record_count": len(records),
                "unique_candidate_count": len(
                    unique_candidates
                ),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", output_path)

    print(
        "\n=== RETRIEVAL COMPLETE ==="
    )


if __name__ == "__main__":
    main()

