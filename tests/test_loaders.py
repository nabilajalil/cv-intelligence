from docx import Document
from cv_intelligence.loaders import load_document

def test_docx(tmp_path):
    path = tmp_path / "resume.docx"
    doc = Document()
    doc.add_paragraph("Python engineer")
    doc.save(path)
    assert "Python engineer" in load_document(path)[0].text
