
from cv_intelligence.bm25_retrieval import tokenize


def test_tokenize():
    assert tokenize("Python, Docker & AWS!") == [
        "python",
        "docker",
        "aws",
    ]
