"""
STAGE 1: INGEST
----------------
Job: load raw documents from data/ and split them into chunks.

Why chunking matters: an LLM can't search your whole document at once,
and embeddings work best on small, focused pieces of text. Too big a
chunk = imprecise retrieval (chunk covers too many topics). Too small
a chunk = not enough context to answer with.

This uses "recursive" chunking: split on paragraph breaks first, then
sentences, then words, only falling back to a smaller separator if a
chunk is still too big. This is the current recommended baseline
(better than naive fixed-character splitting) because it respects the
document's natural structure instead of cutting mid-sentence.
"""

import os
import re
import sys
import unicodedata
from pathlib import Path
from pypdf import PdfReader

# Make stdout tolerant of any unicode character (ligatures, symbols, etc.)
# regardless of the terminal's default encoding - prevents the same
# crash resurfacing on some new character in a future document.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_DIR = Path(__file__).parent.parent / "data"

DOCS_BASE_URL = "https://fastapi.tiangolo.com/"

# FastAPI's markdown headings carry their URL anchor, e.g.
# "## Use `CORSMiddleware` { #use-corsmiddleware }" - capturing these lets
# every chunk cite the exact section of the official docs it came from.
HEADING_RE = re.compile(r"^#{1,6} (.+?) \{ #([\w-]+) \}\s*$", re.MULTILINE)
ANCHOR_RE = re.compile(r" ?\{ #[\w-]+ \}")

CHUNK_SIZE = 800       # characters per chunk (~150-200 tokens) - tune this later
CHUNK_OVERLAP = 100     # characters shared between consecutive chunks,
                         # so an answer that spans a chunk boundary isn't lost


def normalize_text(text: str) -> str:
    """
    Fixes characters that cause trouble downstream: PDF exports (especially
    from academic typesetting like LaTeX) often embed ligatures such as
    'ﬃ' (a single character standing in for 'ffi') instead of plain
    letters. NFKC normalization decomposes these into normal ASCII
    equivalents, which both prevents encoding crashes on some terminals
    and gives the embedding model cleaner, more standard text to work with.
    """
    return unicodedata.normalize("NFKC", text)


def source_url(filename: str) -> str:
    """
    Maps a data/ filename to its page on fastapi.tiangolo.com, following
    the naming convention used when the docs were copied in:
    "tutorial-cors.md" -> /tutorial/cors/, "async.md" -> /async/.
    """
    stem = Path(filename).stem
    if stem == "index":
        return DOCS_BASE_URL
    if stem == "tutorial-index":
        return DOCS_BASE_URL + "tutorial/"
    if stem.startswith("tutorial-"):
        return DOCS_BASE_URL + "tutorial/" + stem[len("tutorial-"):] + "/"
    return DOCS_BASE_URL + stem + "/"


def strip_anchors(text: str) -> str:
    """Removes the "{ #anchor }" markers - for display, not for embedding."""
    return ANCHOR_RE.sub("", text)


def load_documents() -> list[dict]:
    """
    Reads every .txt and .pdf file in data/ and returns a list of
    {"source": filename, "text": full_text} dicts.
    """
    documents = []

    for file_path in DATA_DIR.glob("*"):
        if file_path.suffix in (".txt", ".md"):
            text = file_path.read_text(encoding="utf-8")
            documents.append({"source": file_path.name, "text": normalize_text(text)})

        elif file_path.suffix == ".pdf":
            reader = PdfReader(str(file_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            documents.append({"source": file_path.name, "text": normalize_text(text)})

    if not documents:
        print(f"[!] No .txt or .pdf files found in {DATA_DIR}. Add some documents there first.")

    return documents


def recursive_chunk(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping chunks, preferring to break on paragraph
    breaks, then sentence breaks, then just characters as a last resort.
    """
    separators = ["\n\n", "\n", ". ", " "]  # tried in order, most structural first

    def _split(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        if not seps:
            # No separators left - just hard-cut it
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        sep = seps[0]
        parts = text.split(sep)

        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                # this single part might still be too big - recurse with the next separator
                if len(part) > chunk_size:
                    chunks.extend(_split(part, seps[1:]))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current)
        return chunks

    raw_chunks = _split(text, separators)

    # Add overlap: each chunk gets a tail of the previous chunk prepended,
    # so context isn't lost right at a chunk boundary.
    overlapped = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            overlapped.append(chunk)
        else:
            prev_tail = raw_chunks[i - 1][-overlap:]
            overlapped.append(prev_tail + " " + chunk)

    return overlapped


def ingest_all() -> list[dict]:
    """
    Full ingest pipeline: load documents, chunk each one, and return a
    flat list of {"source": ..., "chunk_id": ..., "text": ...} ready for embedding.
    """
    documents = load_documents()
    all_chunks = []

    for doc in documents:
        text = doc["text"]
        headings = [(m.start(), m.group(1), m.group(2)) for m in HEADING_RE.finditer(text)]
        page_url = source_url(doc["source"])
        search_from = 0

        for i, chunk_text in enumerate(recursive_chunk(text)):
            chunk_text = chunk_text.strip()

            # Locate the chunk in the document (by its tail, since the head
            # carries overlap from the previous chunk), then attribute it
            # to the closest heading at or before its midpoint.
            tail = chunk_text[-80:]
            end = text.find(tail, search_from)
            if end == -1:
                end = text.find(tail)
            if end != -1:
                search_from = end
                end += len(tail)
            midpoint = max(0, end - len(chunk_text) // 2) if end != -1 else 0

            title, anchor = "", ""
            for pos, heading_title, heading_anchor in headings:
                if pos > midpoint:
                    break
                title, anchor = heading_title, heading_anchor

            all_chunks.append({
                "source": doc["source"],
                "chunk_id": f"{doc['source']}::chunk_{i}",
                "text": chunk_text,
                "title": title.replace("`", "") or Path(doc["source"]).stem,
                "url": page_url + (f"#{anchor}" if anchor else ""),
            })

    print(f"[OK] Ingested {len(documents)} document(s) into {len(all_chunks)} chunks.")
    return all_chunks


if __name__ == "__main__":
    # Run this file directly to sanity-check chunking BEFORE embedding anything.
    # This is step 1 of debugging discipline: look at your chunks with your own
    # eyes before trusting them to an embedding model.
    chunks = ingest_all()
    for c in chunks[:3]:
        print("\n---")
        print(f"[{c['chunk_id']}]")
        print(c["text"][:300])
