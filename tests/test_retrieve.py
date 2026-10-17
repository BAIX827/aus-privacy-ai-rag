from privrag.chunk import chunk_section
from privrag.retrieve import app_filter, rrf

from tests.conftest import CHUNKS


def test_rrf_prefers_items_ranked_high_by_both():
    assert rrf([[1, 2, 3], [2, 1, 3]])[:2] in ([1, 2], [2, 1])
    assert rrf([[5], [5], [6]])[0] == 5


def test_app_filter():
    assert app_filter("What does APP 11 require?", CHUNKS) == {1}
    assert app_filter("Can I use ChatGPT?", CHUNKS) is None
    assert app_filter("APP 99", CHUNKS) is None


def test_chunk_overlap():
    parts = chunk_section(" ".join(map(str, range(400))), size=180, overlap=40)
    assert len(parts) == 3 and parts[1].split()[0] == "140"


def test_search_modes(retriever):
    for mode in ("bm25", "dense", "hybrid", "rerank"):
        top = retriever.search("protect personal information security", mode=mode, k=1)[0][0]
        assert top["doc_id"] == "app-11", mode


def test_app_reference_filters_results(retriever):
    hits = retriever.search("APP 6 personal information", k=3)
    assert {c["doc_id"] for c, _ in hits} == {"app-6"}
