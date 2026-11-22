
"""Phase 7.1 — Protected multimodal FastAPI router.

Provides read-only access to the Phase 6C
fingerprint-protected candidate-search pipeline.

No OpenAI API calls.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scripts.demo_multimodal_retrieval import (
    load_profiles,
)
from scripts.demo_multimodal_evidence import (
    search_with_evidence,
)
from scripts.demo_protected_grounded_answers import (
    build_protected_context,
    evaluate_protected_candidate,
)
from scripts.demo_grounded_multimodal_answers import (
    render_answer,
)


router = APIRouter(
    prefix="/multimodal",
    tags=["Protected Multimodal Search"],
)


class ProtectedSearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=500,
    )
    required_skills: list[str] = Field(
        default_factory=list,
        max_length=20,
    )


class VerifiedEvidence(BaseModel):
    field: str
    text: str
    source_file: str
    page_number: int
    decision: str
    integrity_status: str


class ProtectedCandidate(BaseModel):
    candidate_id: str
    eligible: bool
    accepted_evidence: list[VerifiedEvidence]
    withheld_evidence_count: int


class ProtectedSearchResponse(BaseModel):
    query: str
    required_skills: list[str]
    candidate_count: int
    candidates: list[ProtectedCandidate]
    answer: str


@router.post(
    "/protected-search",
    response_model=ProtectedSearchResponse,
)
def protected_search(
    request: ProtectedSearchRequest,
):
    """Return only eligible candidates with verified evidence."""

    try:
        records = load_profiles()
        context = build_protected_context()

        retrieved = search_with_evidence(
            request.query,
            records,
        )

        evaluations = [
            evaluate_protected_candidate(
                candidate,
                context,
                request.required_skills,
            )
            for candidate in retrieved
        ]

        eligible = [
            candidate
            for candidate in evaluations
            if candidate["eligible"]
        ]

        candidates = []

        for candidate in eligible:
            candidates.append(
                ProtectedCandidate(
                    candidate_id=candidate[
                        "candidate_id"
                    ],
                    eligible=True,
                    accepted_evidence=[
                        VerifiedEvidence(
                            field=item["field"],
                            text=item["text"],
                            source_file=item[
                                "source_file"
                            ],
                            page_number=item[
                                "page_number"
                            ],
                            decision=item[
                                "decision"
                            ],
                            integrity_status=item[
                                "integrity_status"
                            ],
                        )
                        for item in candidate[
                            "accepted_evidence"
                        ]
                    ],
                    withheld_evidence_count=len(
                        candidate[
                            "withheld_evidence"
                        ]
                    ),
                )
            )

        return ProtectedSearchResponse(
            query=request.query,
            required_skills=request.required_skills,
            candidate_count=len(candidates),
            candidates=candidates,
            answer=render_answer(
                request.query,
                evaluations,
            ),
        )

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        # Do not expose internal file paths or
        # verification-report details to API clients.
        raise HTTPException(
            status_code=503,
            detail=(
                "Protected multimodal search "
                "is temporarily unavailable."
            ),
        ) from exc
