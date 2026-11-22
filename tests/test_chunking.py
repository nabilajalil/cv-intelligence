import pytest
from cv_intelligence.chunking import chunk_pages
from cv_intelligence.loaders import PageText

def test_section_and_provenance():
    chunks = chunk_pages([PageText(1, "Skills\nPython, Docker\nProjects\nBuilt a service\n")], "doc1", "cand1")
    assert [c.section for c in chunks] == ["skills", "projects"]
    assert all(c.page == 1 and c.document_id == "doc1" for c in chunks)

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_pages([], "d", "c", max_chars=100, overlap=100)
