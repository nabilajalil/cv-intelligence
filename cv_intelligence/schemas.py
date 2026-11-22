from pydantic import BaseModel, Field

class Experience(BaseModel):
    role: str = ""
    employer: str = ""
    dates: str = ""
    responsibilities: list[str] = Field(default_factory=list)

class CandidateProfile(BaseModel):
    name: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    candidate_id: str
    section: str
    page: int | None = None
    text: str
    start_char: int
    end_char: int
