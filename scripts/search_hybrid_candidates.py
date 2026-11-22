
import argparse
import json

from cv_intelligence.hybrid_candidates import (
    search_hybrid_candidates,
)


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid search for distinct CV candidates"
    )

    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    results = search_hybrid_candidates(
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
