
import argparse
import json

from cv_intelligence.bm25_retrieval import search_bm25


def main():
    parser = argparse.ArgumentParser(
        description="BM25 keyword search over CVs"
    )
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    results = search_bm25(args.query, args.limit)

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
