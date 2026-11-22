
import argparse
import json

from cv_intelligence.candidate_retrieval import (
    search_unique_candidates,
)


def main():
    parser = argparse.ArgumentParser(
        description="Search distinct candidates"
    )

    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    results = search_unique_candidates(
        query=args.query,
        limit=args.limit,
    )

    print(
        json.dumps(
            {
                "query": args.query,
                "candidate_count": len(results),
                "candidates": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
