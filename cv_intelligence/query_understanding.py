
from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings


class RecruiterRequirements(BaseModel):
    required_skills: list[str] = Field(
        default_factory=list,
        description="Explicitly required technical skills",
    )


def understand_query(query: str) -> list[str]:
    """Extract technical requirements from a recruiter query."""

    if not query.strip():
        raise ValueError("Recruiter query cannot be empty")

    settings = Settings()

    client = OpenAI(api_key=settings.openai_api_key)

    response = client.beta.chat.completions.parse(
        model=settings.openai_extraction_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Extract technical skills explicitly requested "
                    "by a recruiter. Do not invent requirements. "
                    "Do not treat optional or nice-to-have skills "
                    "as mandatory. Return an empty list if there "
                    "are no explicit technical requirements."
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        response_format=RecruiterRequirements,
    )

    parsed = response.choices[0].message.parsed

    if parsed is None:
        raise ValueError(
            "Could not extract recruiter requirements"
        )

    return list(
        dict.fromkeys(
            skill.strip()
            for skill in parsed.required_skills
            if skill.strip()
        )
    )

