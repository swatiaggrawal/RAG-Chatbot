"""
STAGE 3: RETRIEVE
-------------------
Job: given a user's question, embed it with the SAME embedding model
used for the chunks (Pinecone's hosted llama-text-embed-v2), then ask Pinecone for
the most similar stored vectors.

This file is deliberately runnable on its own (see __main__ below) so
you can sanity-check retrieval quality WITHOUT involving the LLM at
all. This is the single most important debugging habit in RAG: if the
final answer is wrong, check here first. If the right chunk wasn't
retrieved, no amount of prompt tweaking will fix the final answer -
the LLM literally never sees the right information.
"""

import sys
from embed_store import embed, get_index

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def embed_query(query: str) -> list[float]:
    """
    Embeds a single search query. input_type="query" (as opposed to
    "passage" used when storing chunks) tells the model this text is a
    search query, not a document - the embeddings are asymmetric, so
    matching this to embed_store.py's input_type is what makes
    retrieval actually work well.
    """
    return embed([query], input_type="query")[0]


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """
    Returns the top_k most relevant chunks for a given query, each as
    {"text": ..., "source": ..., "distance": ...}.
    Pinecone returns a similarity SCORE (higher = more similar) with
    cosine metric; we convert to a "distance" (lower = more similar)
    so the printed output stays consistent with how this project has
    reported retrieval quality throughout.
    """
    query_embedding = embed_query(query)

    index = get_index()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    retrieved = []
    for match in results.matches:
        retrieved.append({
            "text": match.metadata["text"],
            "source": match.metadata["source"],
            "distance": 1 - match.score,
        })

    return retrieved


if __name__ == "__main__":
    # Manual test loop: type questions, eyeball whether the retrieved
    # chunks actually contain the answer. Do this BEFORE wiring up
    # generation - if this looks wrong, fix chunking/embedding first.
    print("Retrieval test mode. Type a question (or 'quit' to exit).\n")
    while True:
        query = input("Question: ").strip()
        if query.lower() in ("quit", "exit"):
            break

        results = retrieve(query)
        print(f"\nTop {len(results)} chunks:")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] source={r['source']} distance={r['distance']:.3f}")
            print(r["text"][:300])
        print("\n" + "-" * 50 + "\n")
