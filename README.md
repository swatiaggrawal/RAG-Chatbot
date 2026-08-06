# Knowledge Copilot — Grounded RAG Chatbot

A retrieval-augmented chatbot that answers questions about **FastAPI's
official documentation** — every answer is grounded in retrieved source
text and cited, so it never silently makes things up. Built end-to-end:
ingestion, embedding, retrieval, generation, a FastAPI backend, and a
React frontend.

> The document corpus is FastAPI's own docs (MIT-licensed, pulled from
> [tiangolo/fastapi](https://github.com/tiangolo/fastapi)) — a
> real "developer documentation copilot" use case. The architecture
> itself is domain-agnostic: swap the files in `data/` and the same
> pipeline works for internal docs, support knowledge bases, or
> contract Q&A.

**[Live demo →](#)** *[https://rag-chatbot-new.onrender.com](https://rag-chatbot-new.onrender.com)*

## Why this project

Most "RAG chatbot" tutorials stop at a Jupyter notebook. This one is
built as a real, deployed system with a clean separation between four
independently-testable stages — which matters because in RAG, almost
every failure traces back to one specific stage, and if everything's
tangled together you can't tell whether the problem is bad chunking,
bad retrieval, or bad generation.

## Architecture

```
Your documents (.txt, .pdf)
        │
        ▼
   ingest.py        →  recursive chunking (paragraph/sentence-aware,
        │               not naive fixed-character splitting) + Unicode
        │               normalization (fixes PDF ligature artifacts)
        ▼
   embed_store.py    →  Gemini API embeddings, stored in Pinecone
        │               (managed vector DB), batched to respect both
        │               the API's and Pinecone's insert/request limits
        ▼
   retrieve.py       →  dense (embedding) similarity search — given a
        │               question, returns the top-k most relevant chunks
        ▼
   generate.py       →  builds a grounded prompt (context + question),
        │               calls Gemini, returns an answer + which source
        │               documents it actually used
        ▼
   api.py            →  FastAPI wrapper exposing /chat over HTTP
        │
        ▼
   frontend/         →  React chat UI, source citations shown per answer
```

Each pipeline stage (`ingest.py`, `embed_store.py`, `retrieve.py`,
`generate.py`) is also independently runnable from the command line —
useful for debugging retrieval quality without involving the LLM at all.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | Gemini API (`gemini-embedding-001`) | Keeps the deployed backend lightweight — a local model (sentence-transformers + torch) needs 300-500MB+ RAM just to load, which exceeds free-tier hosting limits like Render's 512MB cap |
| Vector store | Pinecone | Managed, serverless vector DB — no infra to provision or persist locally, scales past single-instance disk limits |
| LLM | Gemini (`gemini-3.6-flash`) | Fast, strong free/low-cost tier for iterating on prompts without burning budget |
| Backend | FastAPI | Thin wrapper — no RAG logic lives here, just HTTP plumbing |
| Frontend | React (CDN, no build step) | Full React component model without adding a Node/npm toolchain to the project |

## Key technical decisions

- **Recursive chunking over fixed-size splitting.** Chunks are split on
  paragraph breaks first, falling back to sentences, so a chunk doesn't
  cut off mid-thought — this is now the standard baseline over blind
  character-count chunking.
- **API-based embeddings over a local model, for deployment reasons.**
  This started with local `sentence-transformers` embeddings, which
  work fine for development but pulled in `torch` as a dependency —
  loading that model alone exceeded the 512MB RAM cap on Render's free
  tier. Switched to Gemini's embedding API instead, which removes the
  heavy local dependency entirely at the cost of a small per-batch API
  call. Query-side and document-side embeddings use matched
  `task_type` values (`RETRIEVAL_QUERY` vs `RETRIEVAL_DOCUMENT`), which
  Gemini's asymmetric embeddings expect for good retrieval quality.
- **Pinecone over a local vector store.** Started with Chroma for local
  persistence; moved to Pinecone's managed/serverless offering to avoid
  bundling and persisting a local index file in the deployment
  environment, and to keep the same backend code path working
  identically in dev and production.
- **Grounded generation, not just "stuffed context."** The system prompt
  explicitly instructs the model to answer only from retrieved context
  and say so when it can't — this is what makes it RAG rather than an
  LLM with some text pasted above the question.
- **Source citations returned per answer.** Every response includes which
  document(s) it actually pulled from, so answers are auditable, not a
  black box.
- **Unicode normalization on ingest.** Academic PDFs often embed
  ligatures (e.g. "ﬃ" as one character standing in for "ffi") — these
  get normalized to plain text, which improves both embedding quality
  and cross-platform reliability.
- **Self-indexing on deploy.** The generated vector index isn't
  committed to git, but `data/` (the source docs) is. On startup, the
  API checks if the index is empty and automatically runs the full
  ingest → embed → store pipeline if so — so a fresh deploy is
  self-contained and doesn't need a manual indexing step.

## Setup

```bash
git clone https://github.com/swatiaggrawal/RAG-Chatbot.git
cd RAG-Chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add:
#   GEMINI_API_KEY   (free at https://aistudio.google.com/apikey)
#   PINECONE_API_KEY (free at https://app.pinecone.io)
```

`data/` already contains a curated set of FastAPI's official docs (index,
features, core concepts, and the full tutorial section) — 44 files, ready
to go. Want to swap in your own documents instead? Just replace the
`.txt`/`.md`/`.pdf` files in `data/` and re-run the pipeline below.

```bash
cd src
python ingest.py         # sanity-check chunking
python embed_store.py    # embed + store in Pinecone
python retrieve.py       # test retrieval quality alone
python generate.py       # full end-to-end test (retrieval + LLM answer)
```

## Running the full app

```bash
# Terminal 1 - API
cd src
python api.py            # runs at http://localhost:8000

# Terminal 2 - frontend
cd frontend
python -m http.server 5500   # open http://localhost:5500
```

## Roadmap

- [ ] **v2: hybrid retrieval** — add BM25 keyword search alongside dense
      retrieval, fused via reciprocal rank fusion, since dense-only
      search misses exact identifiers/proper nouns that keyword search
      catches
- [ ] **v2: reranking** — cross-encoder rerank pass on top-k candidates
      for better precision
- [ ] Retrieval evaluation harness (recall@k on a small golden query set)

## License

MIT
