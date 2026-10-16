"""
STAGE 3: RETRIEVE
-------------------
Job: given a question, find the chunks most likely to answer it.

Pipeline (each step is switchable, so the eval harness can measure
what each one is worth - see evals/):

  1. Dense search    - embed the query with the SAME model used for the
                       chunks (llama-text-embed-v2) and ask Pinecone for
                       the nearest vectors. Good at meaning/paraphrase.
  2. Keyword search  - BM25 over the same chunks, in memory (bm25.py).
                       Good at exact identifiers like `jsonable_encoder`.
  3. Fusion          - Reciprocal Rank Fusion merges the two ranked
                       lists. RRF uses only ranks, not raw scores, so it
                       needs no tuning to combine scores on different
                       scales (cosine similarity vs BM25).
  4. Rerank          - an LLM reorders the fused candidates (rerank.py).

This file is runnable on its own (see __main__) to sanity-check
retrieval without involving answer generation at all. If the final
answer is wrong, check here first: if the right chunk wasn't retrieved,
no prompt tweaking will fix it.
"""

import os
import sys
import time

from bm25 import BM25
from embed_store import embed, get_index
from ingest import ingest_all
from rerank import RERANKER, rerank

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RETRIEVAL_MODE = os.environ.get("RETRIEVAL_MODE", "hybrid")  # dense | keyword | hybrid
CANDIDATES = 12  # how many fused candidates the reranker gets to choose from
RRF_K = 60       # standard RRF constant; dampens the weight of top ranks

# The chunk list is rebuilt from data/ at startup (deterministic, so the
# chunk_ids match what's stored in Pinecone). It backs BM25 and gives
# every result its title/url without storing those in Pinecone.
CHUNKS = ingest_all()
CHUNKS_BY_ID = {c["chunk_id"]: c for c in CHUNKS}
KEYWORD_INDEX = BM25([f"{c['title']}\n{c['text']}" for c in CHUNKS])

_index = None


def _get_index():
    global _index
    if _index is None:
        _index = get_index()
    return _index


def embed_query(query: str) -> list[float]:
    return embed([query], input_type="query")[0]


def dense_search(query: str, k: int, query_vector: list[float] | None = None) -> list[dict]:
    vector = query_vector or embed_query(query)
    results = _get_index().query(vector=vector, top_k=k, include_metadata=True)
    found = []
    for match in results.matches:
        chunk = CHUNKS_BY_ID.get(match.id)
        if chunk is None:  # index built from older data/ - fall back to metadata
            chunk = {
                "chunk_id": match.id,
                "source": match.metadata["source"],
                "text": match.metadata["text"],
                "title": match.metadata["source"],
                "url": "",
            }
        found.append(chunk)
    return found


def keyword_search(query: str, k: int) -> list[dict]:
    return [CHUNKS[i] for i, _ in KEYWORD_INDEX.search(query, k)]


def reciprocal_rank_fusion(rankings: list[list[dict]]) -> list[dict]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            scores[chunk["chunk_id"]] = scores.get(chunk["chunk_id"], 0) + 1 / (RRF_K + rank + 1)
    by_id = {c["chunk_id"]: c for ranking in rankings for c in ranking}
    return [by_id[cid] for cid in sorted(scores, key=scores.get, reverse=True)]


def retrieve(
    query: str,
    top_k: int = 5,
    mode: str | None = None,
    reranker: str | None = None,
    query_vector: list[float] | None = None,
    timings: dict | None = None,
    trace: dict | None = None,
) -> list[dict]:
    """
    Returns the top_k chunks for a query, best first. `mode` and
    `reranker` default to the RETRIEVAL_MODE / RERANKER env vars;
    passing `timings` collects per-stage latency in milliseconds, and
    `trace` collects LLM usage for monitoring.
    """
    mode = mode or RETRIEVAL_MODE
    reranker = reranker or RERANKER
    started = time.perf_counter()

    if mode == "dense":
        candidates = dense_search(query, CANDIDATES, query_vector)
    elif mode == "keyword":
        candidates = keyword_search(query, CANDIDATES)
    else:
        candidates = reciprocal_rank_fusion([
            dense_search(query, CANDIDATES, query_vector),
            keyword_search(query, CANDIDATES),
        ])[:CANDIDATES]

    searched = time.perf_counter()
    results = rerank(query, candidates, top_k, method=reranker, trace=trace)

    if timings is not None:
        timings["search_ms"] = round((searched - started) * 1000)
        timings["rerank_ms"] = round((time.perf_counter() - searched) * 1000)
    return results


if __name__ == "__main__":
    # Manual test loop: type questions, eyeball whether the retrieved
    # chunks actually contain the answer. Do this BEFORE judging the
    # generated answers - if this looks wrong, fix retrieval first.
    print(f"Retrieval test mode ({RETRIEVAL_MODE}, reranker={RERANKER}). Type a question (or 'quit').\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break

        timings = {}
        results = retrieve(query, timings=timings)
        print(f"\nTop {len(results)} chunks ({timings}):")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['source']} > {r['title']}")
            print(r["text"][:300])
        print("\n" + "-" * 50 + "\n")
