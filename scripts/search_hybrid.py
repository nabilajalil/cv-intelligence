
import argparse
import json

from cv_intelligence.hybrid_retrieval import search_hybrid


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid CV search using RRF"
    )

    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    results = search_hybrid(
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
