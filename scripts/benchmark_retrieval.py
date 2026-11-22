
from pathlib import Path

from cv_intelligence.advanced_rag.long_context_search import (
    load_cv_documents,
)
from cv_intelligence.advanced_rag.rag_baseline import (
    retrieve_cv_documents,
)
from cv_intelligence.candidate_evaluation import contains_skill


SKILL_QUERIES = [
    ["Python", "Docker", "AWS"],
    ["Python", "LangGraph"],
    ["RAG", "Docker"],
    ["AWS", "Kubernetes"],
    ["Python", "FastAPI"],
]


def evaluate_query(documents: list[dict], skills: list[str], top_k=5):
    query = "Find candidates with " + ", ".join(skills)

    relevant = {
        document["filename"]
        for document in documents
        if all(
            contains_skill(document["text"], skill)
            for skill in skills
        )
    }

    retrieved = retrieve_cv_documents(
        query=query,
        documents=documents,
        top_k=top_k,
    )

    retrieved_names = {
        document["filename"]
        for document in retrieved
    }

    hits = len(relevant & retrieved_names)

    precision = (
        hits / len(retrieved_names)
        if retrieved_names else 0.0
    )

    recall = (
        hits / len(relevant)
        if relevant else None
    )

    return {
        "skills": skills,
        "relevant_count": len(relevant),
        "retrieved_count": len(retrieved_names),
        "hits": hits,
        "precision": precision,
        "recall": recall,
    }


def main():
    documents = load_cv_documents(Path("data/sample_cvs"))

    if not documents:
        raise SystemExit("No CV documents found.")

    print(f"Dataset: {len(documents)} CVs")
    print("Retrieval method: BM25")
    print("Top-k: 5\n")

    results = []

    for skills in SKILL_QUERIES:
        result = evaluate_query(
            documents=documents,
            skills=skills,
            top_k=5,
        )

        results.append(result)

        print("Skills:", ", ".join(skills))
        print("Relevant:", result["relevant_count"])
        print("Hits:", result["hits"])
        print(f"Precision@5: {result['precision']:.2f}")

        if result["recall"] is None:
            print("Recall@5: N/A (no relevant CVs)")
        else:
            print(f"Recall@5: {result['recall']:.2f}")

        print("-" * 40)

    valid_recalls = [
        result["recall"]
        for result in results
        if result["recall"] is not None
    ]

    if valid_recalls:
        mean_recall = sum(valid_recalls) / len(valid_recalls)
        print(f"\nMean Recall@5: {mean_recall:.2f}")
    else:
        print("\nMean Recall@5: N/A")


if __name__ == "__main__":
    main()
