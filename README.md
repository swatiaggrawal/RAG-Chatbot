# Knowledge Copilot — Grounded RAG Chatbot

A retrieval-augmented chatbot that answers questions about **FastAPI's
official documentation**. Every answer is grounded in retrieved source
text, with numbered inline citations that link to the exact section of
the docs, so nothing is silently made up. Built end to end: ingestion,
hybrid retrieval, LLM reranking, conversation memory, streaming
generation, an evaluation harness with CI, a FastAPI backend and a React
frontend.

> The document corpus is FastAPI's own docs (MIT-licensed, pulled from
> [tiangolo/fastapi](https://github.com/tiangolo/fastapi)) — a
> real "developer documentation copilot" use case. The architecture
> itself is domain-agnostic: swap the files in `data/` and the same
> pipeline works for internal docs, support knowledge bases, or
> contract Q&A.

**[Live demo →](https://rag-chatbot-new.onrender.com)** *(free tier: the first request after idling can take ~50s while the server wakes up)*

## Highlights

- **Measured, not guessed.** A 57-question golden set and an eval
  harness score retrieval (hit@k, MRR) per configuration and per
  question type, and an LLM judge grades answers for faithfulness,
  correctness, citations and refusals — with a failure taxonomy that
  says whether a miss came from retrieval or generation. Retrieval
  evals run in GitHub Actions on every push.
- **Hybrid retrieval + reranking.** Dense vectors (Pinecone) and BM25
  keyword search, merged with Reciprocal Rank Fusion, then reranked by
  a fast LLM — each stage justified by the eval numbers below.
- **Conversation memory.** Follow-ups like *"can I add custom headers to
  it?"* are rewritten into standalone search queries using the chat
  history.
- **Streaming answers with clickable citations.** Tokens stream over
  Server-Sent Events; `[1]` markers open the source passage and link to
  the exact docs section.
- **Built for free-tier constraints.** Runs within Render's 512 MB and
  Groq/Pinecone free quotas — several design decisions below came
  directly from measuring those limits.

## Evaluation results

Golden set: 57 hand-written questions in [`evals/golden.jsonl`](evals/golden.jsonl)
— 34 standard, 5 paraphrased (avoid the docs' wording), 5 exact-identifier,
8 multi-turn follow-ups, and 5 the docs can't answer (incl. a
prompt-injection attempt). Full run: 16 Oct 2026.

**Retrieval** (52 answerable questions; did a chunk from the right doc
make the top 5?)

| Configuration | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|
| dense only | 90% | 100% | 0.943 |
| keyword only (BM25) | 79% | 98% | 0.864 |
| hybrid (RRF) | 90% | 100% | 0.952 |
| **hybrid + LLM rerank** | **96%** | **100%** | **0.981** |

Where each stage earns its place (MRR@5 by question type):

| Question type | n | dense | BM25 | hybrid | hybrid + rerank |
|---|---|---|---|---|---|
| standard | 34 | 0.966 | 0.848 | 0.956 | 0.971 |
| paraphrased | 5 | 0.800 | 0.717 | 0.900 | **1.000** |
| follow-ups | 8 | 0.900 | 0.938 | 0.938 | **1.000** |
| exact identifiers | 5 | 1.000 | 1.000 | 1.000 | 1.000 |

**Conversation memory:** searching a follow-up as typed ("can I add
custom headers to *it*?") gets the right doc first 88% of the time;
rewriting it with the chat history first raises that to **100%**.

**Answers** (57 questions, graded by an LLM judge against hand-written
key facts):

| Metric | Score |
|---|---|
| Faithfulness to sources | 4.96 / 5 |
| Correctness vs key facts | 4.85 / 5 |
| Citation accuracy | 94% |
| Correct refusals (unanswerable + injection) | 5 / 5 |
| Overall pass rate | 89% (51 / 57) |
| Latency p50 / p90 (non-streaming, end to end) | 2.1 s / 3.2 s |

**What the failure analysis found.** The 6 failures split into 3
"bad citation" and 3 "incomplete". Reading them showed the citation
failures weren't wrong sources at all: `gpt-oss` sometimes emits its
training-time citation style (`【1】`, `【4†L1-L4】`), which the app didn't
recognize — so those answers rendered with **no clickable citations**
in the UI. Fix: normalize both formats (backend + frontend) and tighten
the prompt. Re-testing the 6 failures afterwards: all 3 citation
failures now pass; 2 "incomplete" answers remain (correct, but skip the
docs' preferred `Annotated` style — linked to code samples missing from
the corpus, see limitations), and one previously passing case refused
on re-run, a reminder that single runs of an LLM pipeline are noisy.

**A production issue the evals surfaced.** The full run exhausted
`gpt-oss-20b`'s 200K tokens/day free quota, and the reranker — by
design — silently fell back to un-reranked order. That would have
quietly degraded the live demo for the rest of the day, so the LLM
layer now switches to a backup model when a model's *daily* quota is
gone, and the eval report flags any run where reranking fell back.

Reproduce with `python evals/run_evals.py --retrieval --generation`
(full report in [`evals/results/`](evals/results/)).

## Architecture

```
                     ┌─────────────── question + recent chat history
                     ▼
   generate.py   rewrite follow-up → standalone query (gpt-oss-20b)
                     │
                     ▼
   retrieve.py   ┌── dense search (Pinecone, llama-text-embed-v2) ──┐
                 │                                                  ├─ RRF fusion → 12 candidates
                 └── keyword search (BM25, in memory) ──────────────┘
                     │
                     ▼
   rerank.py     LLM listwise rerank → top 5 (gpt-oss-20b)
                     │
                     ▼
   generate.py   grounded prompt with numbered sources → streamed
                 answer with [n] citations (gpt-oss-120b)
                     │
                     ▼
   api.py        POST /chat (JSON) · POST /chat/stream (SSE)
                     │
                     ▼
   frontend/     React UI: streaming, citation cards, chat memory

   Offline:  ingest.py (chunking + section anchors) → embed_store.py (Pinecone)
   Quality:  evals/ golden set → run_evals.py → GitHub Actions
```

Each stage is independently runnable from the command line (`python
retrieve.py`, `python generate.py`), so retrieval quality can be checked
without involving the LLM at all.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embeddings | Pinecone Inference (`llama-text-embed-v2`, 1024-dim) | No local model weights, so the backend fits in Render's 512 MB |
| Vector store | Pinecone (serverless) | Managed, persistent, identical in dev and prod |
| Keyword search | BM25, implemented in ~60 lines | ~500 chunks fit in memory; no extra service needed |
| Reranker | Listwise LLM rerank on Groq (`gpt-oss-20b`) | See "Key technical decisions" |
| LLM | Groq (`openai/gpt-oss-120b`) | Fast inference with a free tier |
| Eval judge | Groq (`qwen/qwen3.8-27b`) | Different model family from the answer model, to avoid self-preference |
| Backend | FastAPI + Server-Sent Events | Thin HTTP layer, no RAG logic |
| Frontend | React (CDN, no build step), marked + DOMPurify | No Node toolchain; sanitized Markdown |
| CI | GitHub Actions | Retrieval evals on every push to `main` |

## Key technical decisions

- **Evals before features.** The first eval run showed dense retrieval
  already at ~95% hit@1 on questions phrased like the docs — so the
  golden set was extended with harder slices (paraphrased questions,
  exact-identifier questions, pronoun follow-ups) that reflect real
  users. Results are reported per slice, because an average over easy
  questions hides exactly the cases each technique exists for.
- **Hybrid retrieval with Reciprocal Rank Fusion.** Dense embeddings
  handle paraphrase ("the browser blocks my request" → CORS); BM25
  handles exact identifiers (`call_next`, `Last-Event-ID`). RRF merges
  the two ranked lists using ranks only, so cosine similarities and
  BM25 scores never need to be put on the same scale.
- **An LLM reranker, chosen by measurement.** A local cross-encoder
  (`ms-marco-MiniLM-L-6` via ONNX/fastembed) took **3–6 s per query and
  ~360 MB RAM** on a laptop CPU — unusable on a free-tier server with a
  fraction of a CPU. Pinecone's hosted rerankers cap the free plan at
  **500 requests/month**. A single listwise call to a small, fast LLM on
  Groq (the RankGPT approach) takes ~0.5 s. It falls back to the fused
  order if the call fails, and the eval harness counts those fallbacks
  so a rate limit can't silently skew results.
- **Spreading load across models.** Groq's free tier rate-limits each
  model separately, so answers, reranking/rewriting, and eval judging
  use three different models. Eval runs also cache reranks in-process
  so the generation suite reuses the retrieval suite's calls.
- **Stateless conversation memory.** The client sends the last few
  messages with each request; the server keeps no sessions, so it
  scales and survives restarts trivially. Follow-ups are rewritten into
  standalone queries before search, because "can I add headers to it?"
  retrieves nothing useful on its own.
- **Citations down to the section.** FastAPI's Markdown headings carry
  their URL anchors (`## Use CORSMiddleware { #use-corsmiddleware }`).
  Ingestion attributes each chunk to its nearest heading, so every
  citation links to the exact section on fastapi.tiangolo.com.
- **Grounded generation.** The prompt restricts answers to the numbered
  sources, asks for a kind "not in the docs" when they don't cover the
  question, and treats sources and user messages as data, never
  instructions (a basic prompt-injection guard the eval set tests).
- **Streaming over SSE.** `/chat/stream` sends the sources first (so the
  UI can show what was found), then tokens, then citation and timing
  metadata. Proxy buffering is disabled so tokens arrive live on Render.
- **API-based embeddings, managed vector store.** Local
  `sentence-transformers` pulled in `torch`, which alone exceeded
  Render's 512 MB; Chroma's native index crashed on Render's CPUs.
  Pinecone's hosted embeddings and serverless index remove both issues.
- **Self-indexing on deploy.** On startup the API checks whether the
  index is empty and, if so, runs ingest → embed → store itself, so a
  fresh deploy needs no manual step.

## Known limitations

- The FastAPI docs pull their code samples from separate files via
  include directives (`{* ../../docs_src/... *}`), so the indexed text
  contains the prose but not most of the code. Code in answers is
  limited to what the prose itself shows.
- The golden set is hand-written and small (57 questions); it measures
  relative improvements between configurations well, but absolute
  scores on a larger, user-sourced set would be lower.
- The LLM judge is itself a model and can be wrong; its scores are
  best read as trends across runs, with failures spot-checked by hand.

## Setup

```bash
git clone https://github.com/swatiaggrawal/RAG-Chatbot.git
cd RAG-Chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add:
#   GROQ_API_KEY     (free at https://console.groq.com/keys)
#   PINECONE_API_KEY (free at https://app.pinecone.io)
```

`data/` already contains a curated set of FastAPI's official docs (index,
features, core concepts, and the full tutorial section) — 44 files, ready
to go. To use your own documents, replace the `.txt`/`.md`/`.pdf` files
in `data/`, re-run the pipeline below, and update `evals/golden.jsonl`.

```bash
cd src
python ingest.py         # sanity-check chunking
python embed_store.py    # embed + store in Pinecone
python retrieve.py       # test retrieval alone (hybrid + rerank)
python generate.py       # end-to-end chat in the terminal, with memory
```

## Running the full app

```bash
# Terminal 1 - API
cd src
python api.py                      # http://localhost:8000

# Terminal 2 - frontend
cd frontend
python -m http.server 5500
# open http://localhost:5500/?api=http://localhost:8000
```

## Running the evals

```bash
python evals/run_evals.py --retrieval                 # cheap: search quality only
python evals/run_evals.py --retrieval --generation    # full: adds LLM-judged answers
python evals/run_evals.py --generation --limit 10     # quick check on a subset
```

In CI, add `PINECONE_API_KEY` and `GROQ_API_KEY` under *Settings →
Secrets and variables → Actions*. Retrieval evals run on every push to
`main`; the generation suite can be triggered from the Actions tab.

Retrieval and reranking strategies can be switched without code
changes, e.g. `RETRIEVAL_MODE=dense RERANKER=none` (see `.env.example`).

## License

MIT
