"""
STAGE 2: EMBED + STORE
------------------------
Job: turn each text chunk into a vector (embedding) and store it in
Chroma, a local vector database, so we can later search "which chunks
are semantically similar to this question?"

This is a one-time step (rerun only when your documents change) -
that's why it's a separate file from retrieval, which runs on every
single user query.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

from ingest import ingest_all

# Silence Chroma's usage-telemetry pings - purely cosmetic, has no effect
# on your data, just avoids noisy "Failed to send telemetry event" lines.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

STORAGE_DIR = Path(__file__).parent.parent / "storage"
COLLECTION_NAME = "documents"

# all-MiniLM-L6-v2 is small, fast, and free (runs locally, no API cost) -
# a solid default for a portfolio project. Swap for a bigger model later
# if you want to show you understand the accuracy/speed tradeoff.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chroma's SQLite backend caps how many items can go into a single
# upsert() call. We batch our inserts to stay safely under that limit
# regardless of how many chunks your documents produce.
UPSERT_BATCH_SIZE = 100


def get_chroma_collection():
    """Connects to (or creates) a persistent local Chroma collection."""
    client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def embed_and_store():
    """
    Full pipeline: ingest chunks -> embed them -> store in Chroma.
    Run this once after adding/changing documents in data/.
    """
    chunks = ingest_all()
    if not chunks:
        return

    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection = get_chroma_collection()

    # Chroma stores: an id, the vector, the raw text (as a "document"),
    # and any metadata you want to filter/inspect by later.
    # Inserted in batches so we never exceed Chroma's per-call limit,
    # no matter how many chunks a given document set produces.
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [{"source": c["source"]} for c in chunks]

    for i in range(0, len(chunks), UPSERT_BATCH_SIZE):
        batch_end = i + UPSERT_BATCH_SIZE
        collection.upsert(
            ids=ids[i:batch_end],
            embeddings=embeddings[i:batch_end],
            documents=texts[i:batch_end],
            metadatas=metadatas[i:batch_end],
        )
        print(f"  stored batch {i // UPSERT_BATCH_SIZE + 1} ({min(batch_end, len(chunks))}/{len(chunks)})")

    print(f"[OK] Stored {len(chunks)} chunks in Chroma at {STORAGE_DIR}")


if __name__ == "__main__":
    embed_and_store()
