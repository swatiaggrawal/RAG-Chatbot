"""
STAGE 3b: RERANK
------------------
Job: take the ~12 candidate chunks from hybrid search and reorder them
by how well they actually answer the question, keeping the best few.

First-stage retrieval (embeddings / BM25) scores the query and each
chunk independently, so it's fast but coarse. A reranker reads the
query and the passages together, which is much better at telling
"mentions the same words" apart from "actually answers the question".

Why an LLM reranker? The usual choice is a cross-encoder model, but:
- Running one locally (ms-marco-MiniLM-L-6 via ONNX) was measured at
  3-6 s per query on a laptop CPU and ~360 MB RAM - far too slow on a
  free-tier server with a fraction of a CPU and 512 MB.
- Pinecone's hosted rerankers cap the free plan at 500 requests/month.
So the default ranks passages with a small, fast LLM on Groq in one
"listwise" call (the RankGPT approach). RERANKER=pinecone or none
switch strategies, which the eval harness uses to compare them.
"""

import json
import os
import re

from llm import FAST_MODEL, complete

RERANKER = os.environ.get("RERANKER", "llm")  # llm | pinecone | none
PINECONE_RERANK_MODEL = "bge-reranker-v2-m3"
PASSAGE_CHARS = 500  # enough to judge relevance, keeps the call cheap

RANK_PROMPT = """You rank documentation passages by how useful they are for answering a developer's question about FastAPI.

Read the question and the numbered passages, then return JSON: {"ranking": [passage numbers, most useful first]}.
Put passages that directly answer the question first, then partially relevant ones. Include every passage number exactly once."""

_cache: dict[tuple, list[int]] = {}
# Counted so the eval harness can flag runs where reranking silently fell
# back (e.g. a rate limit), which would otherwise skew its numbers.
stats = {"calls": 0, "failures": 0}


def _llm_order(query: str, chunks: list[dict], trace: dict | None = None) -> list[int]:
    passages = "\n\n".join(
        f"[{i}] ({c['source']} > {c.get('title', '')})\n{c['text'][:PASSAGE_CHARS]}"
        for i, c in enumerate(chunks)
    )
    reply = complete(
        [
            {"role": "system", "content": RANK_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nPassages:\n\n{passages}"},
        ],
        model=FAST_MODEL,
        max_tokens=1024,
        temperature=0,
        json_mode=True,
        purpose="rerank",
        trace=trace,
    )
    try:
        ranking = json.loads(reply)["ranking"]
    except (ValueError, KeyError, TypeError):
        ranking = re.findall(r"\d+", reply)
    return [int(i) for i in ranking]


def _pinecone_order(query: str, chunks: list[dict]) -> list[int]:
    from embed_store import pc

    result = pc.inference.rerank(
        model=PINECONE_RERANK_MODEL,
        query=query,
        documents=[c["text"] for c in chunks],
        top_n=len(chunks),
        return_documents=False,
    )
    return [row.index for row in result.data]


def rerank(
    query: str,
    chunks: list[dict],
    top_n: int,
    method: str | None = None,
    trace: dict | None = None,
) -> list[dict]:
    """
    Returns the top_n chunks in reranked order. If the reranker fails
    (rate limit, malformed output), falls back to the incoming order -
    hybrid search alone still gives a reasonable answer.
    """
    method = method or RERANKER
    if method == "none" or len(chunks) <= 1:
        return chunks[:top_n]

    key = (method, query, tuple(c["chunk_id"] for c in chunks))
    if key not in _cache:
        stats["calls"] += 1
        try:
            order = _llm_order(query, chunks, trace) if method == "llm" else _pinecone_order(query, chunks)
        except Exception as e:
            stats["failures"] += 1
            if trace is not None:
                trace["rerank_fallback"] = True
            print(f"[rerank] {method} reranker failed, keeping hybrid order: {e}")
            return chunks[:top_n]

        # Keep valid, unique indices; append anything the model left out
        # so a sloppy ranking can't drop candidates entirely.
        seen: list[int] = []
        for i in order:
            if 0 <= i < len(chunks) and i not in seen:
                seen.append(i)
        seen += [i for i in range(len(chunks)) if i not in seen]
        if len(_cache) > 256:
            _cache.clear()
        _cache[key] = seen

    return [chunks[i] for i in _cache[key][:top_n]]
