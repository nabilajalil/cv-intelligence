
import argparse
import json

from cv_intelligence.agent import build_agent


def main():
    parser = argparse.ArgumentParser(
        description="Run the CV Intelligence LangGraph agent"
    )

    parser.add_argument("query")

    parser.add_argument(
        "--skills",
        nargs="+",
        default=None,
        help="Optional required skills",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    agent = build_agent()

    result = agent.invoke(
        {
            "query": args.query,
            "required_skills": args.skills or [],
            "limit": args.limit,
            "candidates": [],
            "evaluated_candidates": [],
        }
    )

    print(
        json.dumps(
            {
                "query": result["query"],
                "required_skills": result["required_skills"],
                "candidate_count": len(
                    result["evaluated_candidates"]
                ),
                "candidates": result["evaluated_candidates"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
