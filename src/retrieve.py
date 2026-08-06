"""
STAGE 3: RETRIEVE
-------------------
Job: given a user's question, embed it with the SAME embedding model
used for the chunks (Gemini's embedding API), then ask Chroma for the
most similar chunks.

This file is deliberately runnable on its own (see __main__ below) so
you can sanity-check retrieval quality WITHOUT involving the LLM at
all. This is the single most important debugging habit in RAG: if the
final answer is wrong, check here first. If the right chunk wasn't
retrieved, no amount of prompt tweaking will fix the final answer -
the LLM literally never sees the right information.
"""

import sys
from google.genai import types

from embed_store import get_chroma_collection, client, EMBEDDING_MODEL, EMBEDDING_DIM

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def embed_query(query: str) -> list[float]:
    """
    Embeds a single search query via Gemini. task_type=RETRIEVAL_QUERY
    (as opposed to RETRIEVAL_DOCUMENT used when storing chunks) tells
    the model this text is a search query, not a document - Gemini's
    embeddings are asymmetric, so matching this to embed_store.py's
    task_type is what makes retrieval actually work well.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return response.embeddings[0].values


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """
    Returns the top_k most relevant chunks for a given query, each as
    {"text": ..., "source": ..., "distance": ...}.
    Lower distance = more similar (Chroma default is cosine distance).
    """
    query_embedding = embed_query(query)

    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    retrieved = []
    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": text,
            "source": metadata["source"],
            "distance": distance,
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
