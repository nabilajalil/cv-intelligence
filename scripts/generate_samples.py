from pathlib import Path
from docx import Document

SKILLS = ["Python, LangGraph, RAG, Docker, AWS", "Java, Spring Boot, Kubernetes", "Python, FastAPI, PostgreSQL", "TypeScript, React, Node.js", "Python, NLP, PyTorch"]

def generate(output: Path, count: int = 20):
    output.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        doc = Document()
        doc.add_heading(f"Synthetic Candidate {i+1:02d}", 0)
        doc.add_heading("Summary", 1)
        doc.add_paragraph("Fictional candidate for testing. Not a real person's resume.")
        doc.add_heading("Skills", 1)
        doc.add_paragraph(SKILLS[i % len(SKILLS)])
        doc.add_heading("Experience", 1)
        doc.add_paragraph(f"Software Engineer at Example Company {i%4+1}, 2021–2025. Built and tested production software.")
        doc.add_heading("Projects", 1)
        doc.add_paragraph(f"Project {i+1}: Implemented an application using {SKILLS[i % len(SKILLS)]}.")
        doc.add_heading("Education", 1)
        doc.add_paragraph("BSc Computer Science (fictional)")
        doc.save(output / f"synthetic_cv_{i+1:02d}.docx")

if __name__ == "__main__":
    generate(Path("data/sample_cvs"))
