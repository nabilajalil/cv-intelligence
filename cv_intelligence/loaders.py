from dataclasses import dataclass
from pathlib import Path
import pymupdf
from docx import Document

@dataclass
class PageText:
    page: int
    text: str

def load_document(path: Path) -> list[PageText]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        with pymupdf.open(path) as doc:
            return [PageText(i + 1, page.get_text()) for i, page in enumerate(doc)]
    if ext == ".docx":
        doc = Document(path)
        lines = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                lines.append(" | ".join(cell.text for cell in row.cells))
        return [PageText(1, "\n".join(lines))]
    raise ValueError("Only PDF and DOCX are supported")
