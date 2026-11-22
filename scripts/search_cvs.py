
import argparse
import json

from cv_intelligence.retrieval import search_candidates


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search over indexed CVs"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Natural-language search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of chunks to return",
    )

    args = parser.parse_args()

    results = search_candidates(
        query=args.query,
        limit=args.limit,
    )

    print(
        json.dumps(
            {
                "query": args.query,
                "result_count": len(results),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
