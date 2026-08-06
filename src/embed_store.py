"""
STAGE 2: EMBED + STORE
------------------------
Job: turn each text chunk into a vector (embedding) and store it in
Chroma, a local vector database, so we can later search "which chunks
are semantically similar to this question?"

Embeddings are generated via the Gemini API rather than a local model.
This is a deliberate deployment tradeoff: a local sentence-transformers
model (and the torch library it depends on) needs several hundred MB
of RAM just to load, which exceeds free-tier hosting limits (e.g.
Render's 512MB cap). Calling Gemini's embedding API instead keeps the
deployed service lightweight, at the cost of an API call per chunk
batch instead of local computation - a reasonable tradeoff for a
project this size, and a good example of a real cost/latency/memory
tradeoff you'd be expected to reason about in production RAG systems.

This is a one-time step (rerun only when your documents change) -
that's why it's a separate file from retrieval, which runs on every
single user query.
"""

import os
import time
import chromadb
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path

from ingest import ingest_all

load_dotenv()

# Silence Chroma's usage-telemetry pings - purely cosmetic, has no effect
# on your data, just avoids noisy "Failed to send telemetry event" lines.
os.environ["ANONYMIZED_TELEMETRY"] = "False"

STORAGE_DIR = Path(__file__).parent.parent / "storage"
COLLECTION_NAME = "documents"

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
# 768 dimensions is Google's suggested size for retrieval use cases -
# smaller than the 3072-dim default, which keeps Chroma's storage and
# query speed reasonable without a meaningful quality loss for this use case.
EMBEDDING_DIM = 768

# Gemini's embed_content endpoint caps input batches at 100 strings per call.
EMBED_BATCH_SIZE = 100

# Chroma's SQLite backend caps how many items can go into a single
# upsert() call. We batch our inserts to stay safely under that limit
# regardless of how many chunks your documents produce.
UPSERT_BATCH_SIZE = 100


def get_chroma_collection():
    """Connects to (or creates) a persistent local Chroma collection."""
    chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts via the Gemini API, batching to stay under
    the API's per-call limit. task_type=RETRIEVAL_DOCUMENT tells the
    model these are documents to be searched (as opposed to a query
    searching them) - Gemini's embeddings are asymmetric, so matching
    this correctly on both the storage side (here) and the query side
    (retrieve.py) meaningfully improves retrieval quality.

    Includes retry-with-backoff on rate limit (429) errors, since free
    or lightly-provisioned API tiers can hit per-minute request caps -
    this waits and retries instead of crashing the whole pipeline.
    """
    all_embeddings = []
    max_retries = 5

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]

        for attempt in range(max_retries):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=EMBEDDING_DIM,
                    ),
                )
                all_embeddings.extend([e.values for e in response.embeddings])
                print(f"  embedded batch {i // EMBED_BATCH_SIZE + 1} ({min(i + EMBED_BATCH_SIZE, len(texts))}/{len(texts)})")
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    wait_time = 20 * (attempt + 1)  # 20s, 40s, 60s, 80s, 100s
                    print(f"  Rate limited - waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise  # not a rate-limit error - don't hide it, let it fail loudly
        else:
            raise RuntimeError(
                f"Failed to embed batch starting at index {i} after {max_retries} retries. "
                "This usually means the rate limit is a hard daily cap, not a per-minute one - "
                "check your quota/billing at https://ai.dev/rate-limit"
            )

        time.sleep(2)  # pause between successful batches to stay under per-minute limits

    return all_embeddings


def embed_and_store():
    """
    Full pipeline: ingest chunks -> embed them via Gemini -> store in Chroma.
    Run this once after adding/changing documents in data/.
    """
    chunks = ingest_all()
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks via Gemini API...")
    embeddings = embed_texts(texts)

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
