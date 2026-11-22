from pathlib import Path
import chromadb
from openai import OpenAI
from .schemas import Chunk

class VectorStore:
    def __init__(self, path: Path, model: str = "text-embedding-3-small", client: OpenAI | None = None):
        self.client = client or OpenAI()
        self.model = model
        self.collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection("cv_chunks")

    def upsert(self, chunks: list[Chunk]):
        if not chunks:
            return
        vectors = self.client.embeddings.create(model=self.model, input=[c.text for c in chunks]).data
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[v.embedding for v in vectors],
            documents=[c.text for c in chunks],
            metadatas=[{"document_id": c.document_id, "candidate_id": c.candidate_id, "section": c.section, "page": c.page or 0, "start_char": c.start_char, "end_char": c.end_char} for c in chunks],
        )

    def search(self, query: str, limit: int = 5):
        embedding = self.client.embeddings.create(model=self.model, input=[query]).data[0].embedding
        return self.collection.query(query_embeddings=[embedding], n_results=limit)

    
    def get_candidate_chunks(self, candidate_id: str) -> list[dict]:
        """Return all indexed chunks belonging to one candidate."""

        data = self.collection.get(
            where={"candidate_id": candidate_id},
            include=["documents", "metadatas"],
        )

        results = []

        for chunk_id, text, metadata in zip(
            data["ids"],
            data["documents"],
            data["metadatas"],
        ):
            results.append(
                {
                    "chunk_id": chunk_id,
                    "candidate_id": metadata["candidate_id"],
                    "document_id": metadata["document_id"],
                    "section": metadata["section"],
                    "page": metadata["page"],
                    "text": text,
                }
            )

        return results
