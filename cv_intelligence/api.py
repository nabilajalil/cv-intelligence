
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import build_agent
from .config import Settings

from .protected_api import router as protected_router


app = FastAPI(
    title="CV Intelligence API",
    description="Evidence-grounded candidate retrieval and evaluation",
    version="0.2.0",
)

app.include_router(protected_router)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    required_skills: list[str] = Field(default_factory=list)
    limit: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    query: str
    required_skills: list[str]
    candidate_count: int
    candidates: list[dict]


def load_candidate_names() -> dict[str, dict]:
    """Read candidate names and filenames from saved profiles."""
    settings = Settings()
    profile_dir = Path(settings.cv_data_dir) / "profiles"

    candidates = {}

    for path in profile_dir.glob("*.json"):
        try:
            with path.open(encoding="utf-8") as file:
                data = json.load(file)

            candidate_id = data.get("candidate_id")
            profile = data.get("profile", {})

            if not candidate_id or not isinstance(profile, dict):
                continue

            candidates[candidate_id] = {
                "candidate_name": profile.get("name") or "Unknown candidate",
                "filename": data.get("filename", ""),
            }

        except (OSError, ValueError, TypeError):
            continue

    return candidates


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search_candidates(request: SearchRequest):
    try:
        agent = build_agent()

        result = agent.invoke(
            {
                "query": request.query,
                "required_skills": request.required_skills,
                "limit": request.limit,
                "candidates": [],
                "evaluated_candidates": [],
            }
        )

        candidate_names = load_candidate_names()
        candidates = []

        for candidate in result["evaluated_candidates"]:
            candidate_id = candidate["candidate_id"]

            details = candidate_names.get(
                candidate_id,
                {
                    "candidate_name": "Unknown candidate",
                    "filename": "",
                },
            )

            enriched = {
                **candidate,
                **details,
            }

            candidates.append(enriched)

        return SearchResponse(
            query=result["query"],
            required_skills=result["required_skills"],
            candidate_count=len(candidates),
            candidates=candidates,
        )

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
