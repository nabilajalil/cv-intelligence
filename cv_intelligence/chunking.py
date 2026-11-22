import hashlib
import re
from .schemas import Chunk
from .loaders import PageText

HEADINGS = re.compile(r"^(summary|profile|skills|technical skills|experience|work experience|education|projects|certifications)\s*:?$", re.I)

def chunk_pages(pages: list[PageText], document_id: str, candidate_id: str, max_chars: int = 1400, overlap: int = 180) -> list[Chunk]:
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Invalid chunk size or overlap")
    chunks = []
    for page in pages:
        section = "general"
        blocks = []
        offset = 0
        for line in page.text.splitlines(keepends=True):
            stripped = line.strip()
            if HEADINGS.fullmatch(stripped):
                if blocks:
                    chunks.extend(_split_blocks(blocks, section, page.page, document_id, candidate_id, max_chars, overlap))
                    blocks = []
                section = stripped.rstrip(":").lower()
            if stripped:
                blocks.append((offset, line))
            offset += len(line)
        if blocks:
            chunks.extend(_split_blocks(blocks, section, page.page, document_id, candidate_id, max_chars, overlap))
    return chunks

def _split_blocks(blocks, section, page, document_id, candidate_id, max_chars, overlap):
    text = "".join(line for _, line in blocks)
    base = blocks[0][0]
    result = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + max_chars // 2, end)
            if boundary > start:
                end = boundary + 1
        excerpt = text[start:end].strip()
        if excerpt:
            digest = hashlib.sha256(f"{document_id}:{page}:{base+start}:{excerpt}".encode()).hexdigest()[:20]
            result.append(Chunk(chunk_id=digest, document_id=document_id, candidate_id=candidate_id, section=section, page=page, text=excerpt, start_char=base+start, end_char=base+end))
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return result
