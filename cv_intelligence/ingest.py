import hashlib
import json
from pathlib import Path
from .loaders import load_document
from .chunking import chunk_pages
from .extraction import extract_profile
from .store import VectorStore
from .config import Settings
from openai import OpenAI

def ingest(path: Path, settings: Settings | None = None) -> dict:
    settings = settings or Settings()
    if path.suffix.lower() not in {".pdf", ".docx"}:
        raise ValueError("Unsupported file extension")
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("File exceeds 10 MB limit")
    data = path.read_bytes()
    document_id = hashlib.sha256(data).hexdigest()
    candidate_id = document_id[:16]  # MVP: one document per candidate
    pages = load_document(path)
    text = "\n".join(page.text for page in pages)
    if not text.strip():
        raise ValueError("No extractable text; scanned PDFs are not supported in MVP")
    if not settings.openai_api_key or settings.openai_api_key == "replace-me":
        raise ValueError("Set OPENAI_API_KEY in .env before ingestion")
    openai_client = OpenAI(api_key=settings.openai_api_key)
    profile = extract_profile(text, settings.openai_extraction_model, client=openai_client)
    chunks = chunk_pages(pages, document_id, candidate_id)
    VectorStore(settings.cv_data_dir / "chroma", settings.openai_embedding_model, client=openai_client).upsert(chunks)
    profiles = settings.cv_data_dir / "profiles"
    profiles.mkdir(parents=True, exist_ok=True)
    (profiles / f"{document_id}.json").write_text(json.dumps({"document_id": document_id, "candidate_id": candidate_id, "filename": path.name, "profile": profile.model_dump(), "chunk_count": len(chunks)}, indent=2))
    return {"document_id": document_id, "candidate_id": candidate_id, "chunk_count": len(chunks), "name": profile.name}
