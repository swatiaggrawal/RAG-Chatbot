"""
STAGE 3: RETRIEVE
-------------------
Job: given a user's question, embed it with the SAME model used for
the chunks, then ask Chroma for the most similar chunks.

This file is deliberately runnable on its own (see __main__ below) so
you can sanity-check retrieval quality WITHOUT involving the LLM at
all. This is the single most important debugging habit in RAG: if the
final answer is wrong, check here first. If the right chunk wasn't
retrieved, no amount of prompt tweaking will fix the final answer -
the LLM literally never sees the right information.
"""

import sys
from sentence_transformers import SentenceTransformer

from embed_store import get_chroma_collection, EMBEDDING_MODEL

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Cache the model so we don't reload it on every call
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def retrieve(query: str, top_k: int = 4) -> list[dict]:
    """
    Returns the top_k most relevant chunks for a given query, each as
    {"text": ..., "source": ..., "distance": ...}.
    Lower distance = more similar (Chroma default is cosine distance).
    """
    model = _get_model()
    query_embedding = model.encode([query]).tolist()

    collection = get_chroma_collection()
    results = collection.query(
        query_embeddings=query_embedding,
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
