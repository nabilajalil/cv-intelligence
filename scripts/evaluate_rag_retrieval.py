
import argparse

from pathlib import Path

from cv_intelligence.advanced_rag.long_context_search import (
    load_cv_documents,
)
from cv_intelligence.advanced_rag.rag_baseline import (
    retrieve_cv_documents,
)
from cv_intelligence.candidate_evaluation import contains_skill


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--skills",
        nargs="+",
        default=["Python", "Docker", "AWS"],
    )

    parser.add_argument(
        "--limit-documents",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    documents = load_cv_documents(
        Path("data/sample_cvs")
    )[:args.limit_documents]

    if not documents:
        raise SystemExit("No CV documents found.")

    if args.top_k < 1:
        raise SystemExit("--top-k must be at least 1.")

    query = "Find developers with " + ", ".join(args.skills)

    # Provisional ground truth:
    # Every required skill must appear explicitly in the CV.
    relevant = {
        document["filename"]
        for document in documents
        if all(
            contains_skill(document["text"], skill)
            for skill in args.skills
        )
    }

    retrieved = retrieve_cv_documents(
        query=query,
        documents=documents,
        top_k=args.top_k,
    )

    retrieved_names = [
        document["filename"]
        for document in retrieved
    ]

    hits = len(relevant.intersection(retrieved_names))

    precision = (
        hits / len(retrieved_names)
        if retrieved_names else 0.0
    )

    recall = (
        hits / len(relevant)
        if relevant else None
    )

    print("Required skills:", args.skills)
    print("Documents evaluated:", len(documents))
    print("Relevant CVs:", sorted(relevant))
    print("Retrieved CVs:", retrieved_names)
    print("Relevant retrieved:", hits)

    print(f"Precision@{args.top_k}: {precision:.2f}")

    if recall is None:
        print(
            f"Recall@{args.top_k}: N/A "
            "(no relevant CVs in this dataset)"
        )
    else:
        print(f"Recall@{args.top_k}: {recall:.2f}")


if __name__ == "__main__":
    main()

