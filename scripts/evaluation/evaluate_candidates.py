
import argparse
import json

from cv_intelligence.candidate_evaluation import (
    evaluate_candidates,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate explicit skills in retrieved CV evidence"
    )

    parser.add_argument("query")
    parser.add_argument(
        "--skills",
        nargs="+",
        required=True,
        help="Required skills, separated by spaces",
    )
    parser.add_argument("--limit", type=int, default=5)

    args = parser.parse_args()

    results = evaluate_candidates(
        query=args.query,
        required_skills=args.skills,
        limit=args.limit,
    )

    print(json.dumps(
        {
            "query": args.query,
            "required_skills": args.skills,
            "candidates": results,
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()

