"""
FastAPI wrapper around the RAG pipeline.

This file doesn't contain the core RAG logic - it exposes
generate.py's answer_question() as an HTTP endpoint. The one exception
is the startup auto-indexing logic below: since the deployed backend's
disk doesn't persist your local Chroma database (it's excluded from
git), the app checks on startup whether any documents are indexed yet,
and if not, runs the full ingest -> embed -> store pipeline itself
before accepting requests. This makes the deployment self-contained:
push the code + data/, and it builds its own search index on first boot.
"""

import json
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from generate import answer_question, stream_answer
from guardrails import rate_limiter
from monitoring import metrics
from embed_store import get_index, embed_and_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: check if the Pinecone index already has vectors stored.
    # This matters for deployment - a fresh backend instance has no
    # local database (Pinecone is external/persistent), but on a
    # brand new index there's nothing indexed yet, so this builds it
    # automatically on first boot.
    index = get_index()
    stats = index.describe_index_stats()
    vector_count = stats.get("total_vector_count", 0)

    if vector_count == 0:
        print("No indexed documents found - running ingest + embed on startup...")
        embed_and_store()
    else:
        print(f"Found {vector_count} already-indexed chunks - skipping re-index.")
    yield
    # (nothing needed on shutdown)


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)


def elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)

# CORS: allows your frontend (running on a different port/domain) to
# call this API from the browser. Wide open for local dev; when you
# deploy, narrow allow_origins to your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatRequest(BaseModel):
    # Length limits are a cheap guardrail: they cap how many tokens one
    # request can spend and how much text can be stuffed into the prompt.
    question: str = Field(min_length=1, max_length=2000)
    # Recent conversation, sent by the client - the server keeps no
    # session state, so it scales and survives restarts trivially.
    history: list[Message] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    id: int
    source: str
    title: str
    url: str
    snippet: str


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    sources: list[Source]
    cited: list[int]
    search_query: str
    timings: dict[str, int]


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]


def client_ip(request: Request) -> str:
    # Behind Render's proxy the real client is the first X-Forwarded-For entry.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(request: Request) -> None:
    retry_after = rate_limiter.check(client_ip(request))
    if retry_after is not None:
        metrics.note_rate_limited()
        raise HTTPException(
            status_code=429,
            detail=f"You're sending questions a bit fast. Please try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


@app.get("/health")
def health():
    """Simple check that the API is up - useful once deployed, to
    confirm the server is running before debugging the actual chat."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
def chat(request: ChatRequest):
    """
    Main endpoint: takes a question, runs it through the RAG pipeline
    (retrieve + generate), returns the answer and which source
    documents it was grounded in.
    """
    request_id = metrics.new_request_id()
    history = [m.model_dump() for m in request.history]
    trace: dict = {}
    started = time.perf_counter()
    try:
        result = answer_question(request.question, history, trace=trace)
    except Exception as e:
        # HTTPException (unlike an unhandled crash) goes through the CORS
        # middleware, so the browser can actually read this error.
        print(f"[chat] failed: {e!r}")
        metrics.record(request_id, "/chat", trace, {}, elapsed_ms(started), error=type(e).__name__)
        raise HTTPException(status_code=502, detail="The answer service failed. Please try again.")

    metrics.record(request_id, "/chat", trace, result["timings"], elapsed_ms(started),
                   len(result["sources"]), len(result["cited"]))
    return ChatResponse(request_id=request_id, **{k: result[k] for k in ChatResponse.model_fields if k in result})


@app.post("/chat/stream", dependencies=[Depends(enforce_rate_limit)])
def chat_stream(request: ChatRequest):
    """
    Same pipeline as /chat, streamed as Server-Sent Events so the answer
    appears as it's generated. Events, in order:
      meta  - request id, sources found, the (possibly rewritten) search query
      token - a piece of answer text (many of these)
      done  - which sources were cited + per-stage timings
      error - if something failed midway
    """
    request_id = metrics.new_request_id()
    history = [m.model_dump() for m in request.history]

    def events():
        trace: dict = {}
        started = time.perf_counter()
        timings, n_sources, n_cited, error = {}, 0, 0, None
        try:
            for event, data in stream_answer(request.question, history, trace=trace):
                if event == "meta":
                    data = {"request_id": request_id, **data}
                    n_sources = len(data["sources"])
                elif event == "done":
                    timings, n_cited = data["timings"], len(data["cited"])
                yield f"event: {event}\ndata: {json.dumps(data)}\n\n"
        except Exception as e:
            print(f"[chat/stream] failed: {e!r}")
            error = type(e).__name__
            message = "The answer service failed. Please try again."
            yield f"event: error\ndata: {json.dumps({'message': message})}\n\n"
        finally:
            metrics.record(request_id, "/chat/stream", trace, timings, elapsed_ms(started),
                           n_sources, n_cited, error)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Stop proxies (e.g. Render's) from buffering the stream into one blob.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/feedback")
def feedback(body: FeedbackRequest):
    """Thumbs up/down on an answer, linked to its request's metrics."""
    if not metrics.add_feedback(body.request_id, body.rating):
        raise HTTPException(status_code=404, detail="Unknown or expired request id.")
    return {"ok": True}


@app.get("/stats")
def stats():
    """
    Aggregate health of the last 24 hours: traffic, latency per stage,
    citation and fallback rates, user satisfaction and token usage
    against the free-tier daily quota. Powers frontend/stats.html.
    """
    return metrics.stats()


if __name__ == "__main__":
    # Run with: python api.py
    # (equivalent to: uvicorn api:app --reload --port 8000)
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
