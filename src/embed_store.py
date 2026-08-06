"""
STAGE 2: EMBED + STORE
------------------------
Job: turn each text chunk into a vector (embedding) and store it in
Pinecone, a managed cloud vector database, so we can later search
"which chunks are semantically similar to this question?"

Using a managed vector DB (rather than a local one like Chroma) is a
deliberate deployment choice: Chroma's local index depends on
chroma-hnswlib, a compiled C++ extension that crashed with an "illegal
instruction" error on Render's free-tier CPU hardware (a real,
reproducible compatibility issue, not a bug in this code). Because
Pinecone's actual vector search runs on Pinecone's own servers, the
Render container only ever makes plain HTTP calls to it - removing
that whole class of native-dependency crash.

Embeddings themselves still come from the Gemini API rather than a
local model, for the same memory reason as before (no torch/local
model weights needed on the deployed backend).

This is a one-time step (rerun only when your documents change) -
that's why it's a separate file from retrieval, which runs on every
single user query.
"""

import os
import time
from pinecone import Pinecone, ServerlessSpec
from google import genai
from google.genai import types
from dotenv import load_dotenv

from ingest import ingest_all

load_dotenv()

genai_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
# 768 dimensions is Google's suggested size for retrieval use cases -
# this MUST match the index's configured dimension below.
EMBEDDING_DIM = 768

INDEX_NAME = "rag-chatbot-fastapi-docs"

# Gemini's embed_content endpoint caps input batches at 100 strings per call.
EMBED_BATCH_SIZE = 100
# Pinecone recommends keeping upsert batches in the 100-500 range for
# a good balance of throughput and reliability.
UPSERT_BATCH_SIZE = 100


def get_index():
    """
    Connects to the Pinecone index, creating it first if it doesn't
    exist yet. Index creation is a one-time thing (free tier allows
    only one index) - this function is safe to call repeatedly, since
    it only creates the index if it's genuinely missing.
    """
    if not pc.has_index(INDEX_NAME):
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        # Index creation is async on Pinecone's side - wait until it's
        # actually ready before trying to use it.
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print("  waiting for index to be ready...")
            time.sleep(2)

    return pc.Index(INDEX_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embeds a list of texts via the Gemini API, batching to stay under
    the API's per-call limit. task_type=RETRIEVAL_DOCUMENT tells the
    model these are documents to be searched (as opposed to a query
    searching them) - Gemini's embeddings are asymmetric, so matching
    this correctly on both the storage side (here) and the query side
    (retrieve.py) meaningfully improves retrieval quality.

    Includes retry-with-backoff on rate limit (429) errors.
    """
    all_embeddings = []
    max_retries = 5

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]

        for attempt in range(max_retries):
            try:
                response = genai_client.models.embed_content(
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
                    wait_time = 20 * (attempt + 1)
                    print(f"  Rate limited - waiting {wait_time}s before retry ({attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise
        else:
            raise RuntimeError(
                f"Failed to embed batch starting at index {i} after {max_retries} retries. "
                "Check your quota/billing at https://ai.dev/rate-limit"
            )

        time.sleep(2)

    return all_embeddings


def embed_and_store():
    """
    Full pipeline: ingest chunks -> embed them via Gemini -> upsert
    into Pinecone. Run this once after adding/changing documents in
    data/. Pinecone stores the vector + metadata (source, text)
    together, so retrieval can pull the actual chunk text back out
    without a separate lookup.
    """
    chunks = ingest_all()
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks via Gemini API...")
    embeddings = embed_texts(texts)

    index = get_index()

    vectors = [
        (
            chunks[i]["chunk_id"],
            embeddings[i],
            {"source": chunks[i]["source"], "text": chunks[i]["text"]},
        )
        for i in range(len(chunks))
    ]

    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i:i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"  stored batch {i // UPSERT_BATCH_SIZE + 1} ({min(i + UPSERT_BATCH_SIZE, len(vectors))}/{len(vectors)})")

    print(f"[OK] Stored {len(chunks)} chunks in Pinecone index '{INDEX_NAME}'")


if __name__ == "__main__":
    embed_and_store()
