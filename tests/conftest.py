"""
Shared test setup. Unit tests never touch the network: Groq, Pinecone
and the Prompt Guard classifier are replaced with stubs per test, so the
suite runs in seconds, needs no API keys, and costs no quota in CI.
"""

import os
import sys
from pathlib import Path

import pytest

# Dummy keys so the SDK clients can be constructed; they're never used.
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(autouse=True)
def isolate_caches():
    """Clear module-level caches and quota state between tests."""
    import generate
    import llm
    import rerank

    rerank._cache.clear()
    rerank.stats.update(calls=0, failures=0)
    generate._rewrite_cache.clear()
    llm._exhausted_until.clear()
    yield


def make_chunk(chunk_id: str, source: str = "doc.md", text: str = "some text", title: str = "Title") -> dict:
    return {"chunk_id": chunk_id, "source": source, "text": text, "title": title,
            "url": f"https://example.com/{source}#{chunk_id}"}
