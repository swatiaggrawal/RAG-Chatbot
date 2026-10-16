"""Keyword search, chunk metadata, rank fusion and reranking - all offline."""

import pytest

import ingest
import rerank
import retrieve
from bm25 import BM25, tokenize
from conftest import make_chunk


# ----------------------------------------------------------------- BM25

def test_tokenize_keeps_identifiers_and_drops_stopwords():
    tokens = tokenize("How do I use allow_origins with the CORS middleware?")
    assert "allow_origins" in tokens            # whole identifier kept
    assert {"allow", "origins", "cors", "middleware"} <= set(tokens)
    assert not {"how", "do", "i", "the", "with"} & set(tokens)


def test_bm25_ranks_rare_exact_term_first():
    docs = [
        "FastAPI path operations and FastAPI routing",
        "Use jsonable_encoder to convert models for FastAPI",
        "FastAPI request bodies with Pydantic",
    ]
    results = BM25(docs).search("what does jsonable_encoder do", k=3)
    assert results[0][0] == 1
    assert all(score > 0 for _, score in results)


def test_bm25_returns_nothing_for_unknown_terms_and_respects_k():
    index = BM25(["alpha beta", "beta gamma", "gamma delta"])
    assert index.search("zeta", k=5) == []
    assert len(index.search("beta gamma", k=1)) == 1


# --------------------------------------------------------------- ingest

@pytest.mark.parametrize("filename, url", [
    ("index.md", "https://fastapi.tiangolo.com/"),
    ("tutorial-index.md", "https://fastapi.tiangolo.com/tutorial/"),
    ("tutorial-cors.md", "https://fastapi.tiangolo.com/tutorial/cors/"),
    ("async.md", "https://fastapi.tiangolo.com/async/"),
])
def test_source_url(filename, url):
    assert ingest.source_url(filename) == url


def test_strip_anchors():
    assert ingest.strip_anchors("## Use CORSMiddleware { #use-corsmiddleware }") == "## Use CORSMiddleware"


def test_chunks_carry_section_title_and_deep_link():
    chunks = retrieve.CHUNKS
    assert chunks and all(c["title"] and c["url"].startswith("https://fastapi.tiangolo.com/") for c in chunks)
    cors = next(c for c in chunks if c["source"] == "tutorial-cors.md" and "CORSMiddleware" in c["text"])
    assert cors["url"].startswith("https://fastapi.tiangolo.com/tutorial/cors/#")


# --------------------------------------------------------------- fusion

def test_rrf_prefers_chunks_ranked_well_by_both_lists():
    a, b, c, d = (make_chunk(x) for x in "abcd")
    fused = retrieve.reciprocal_rank_fusion([[a, b, c], [b, d, a]])
    assert [x["chunk_id"] for x in fused][:2] == ["b", "a"]
    assert {x["chunk_id"] for x in fused} == {"a", "b", "c", "d"}


def test_keyword_retrieval_end_to_end_offline():
    timings = {}
    results = retrieve.retrieve("jsonable_encoder", top_k=3, mode="keyword", reranker="none", timings=timings)
    assert len(results) == 3
    assert results[0]["source"] == "tutorial-encoder.md"
    assert {"search_ms", "rerank_ms"} <= timings.keys()


# --------------------------------------------------------------- rerank

CHUNKS = [make_chunk(str(i)) for i in range(5)]


def test_rerank_applies_model_order_and_repairs_it(monkeypatch):
    # Out-of-range, duplicate and missing indices must not lose candidates.
    monkeypatch.setattr(rerank, "_llm_order", lambda q, c, trace=None: [3, 3, 9, 1])
    result = rerank.rerank("q", CHUNKS, top_n=5, method="llm")
    assert [c["chunk_id"] for c in result] == ["3", "1", "0", "2", "4"]


def test_rerank_falls_back_to_incoming_order_on_failure(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(rerank, "_llm_order", boom)
    trace = {}
    result = rerank.rerank("q", CHUNKS, top_n=2, method="llm", trace=trace)
    assert [c["chunk_id"] for c in result] == ["0", "1"]
    assert rerank.stats == {"calls": 1, "failures": 1}
    assert trace["rerank_fallback"] is True


def test_rerank_caches_and_none_method_skips_the_model(monkeypatch):
    calls = []
    monkeypatch.setattr(rerank, "_llm_order", lambda q, c, trace=None: calls.append(q) or [4, 0])
    rerank.rerank("q", CHUNKS, top_n=2, method="llm")
    rerank.rerank("q", CHUNKS, top_n=2, method="llm")
    assert len(calls) == 1
    assert rerank.rerank("other", CHUNKS, top_n=2, method="none") == CHUNKS[:2]
    assert len(calls) == 1
