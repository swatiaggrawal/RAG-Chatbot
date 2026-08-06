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

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from generate import answer_question
from embed_store import get_chroma_collection, embed_and_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: check if the vector database already has documents indexed.
    # This matters for deployment - a fresh Render instance has no
    # storage/ data (it's gitignored), so without this check every
    # /chat request would fail with "no documents indexed."
    collection = get_chroma_collection()
    if collection.count() == 0:
        print("No indexed documents found - running ingest + embed on startup...")
        embed_and_store()
    else:
        print(f"Found {collection.count()} already-indexed chunks - skipping re-index.")
    yield
    # (nothing needed on shutdown)


app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)

# CORS: allows your frontend (running on a different port/domain) to
# call this API from the browser. Wide open for local dev; when you
# deploy, narrow allow_origins to your actual frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    """Simple check that the API is up - useful once deployed, to
    confirm the server is running before debugging the actual chat."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main endpoint: takes a question, runs it through the RAG pipeline
    (retrieve + generate), returns the answer and which source
    documents it was grounded in.
    """
    result = answer_question(request.question)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


if __name__ == "__main__":
    # Run with: python api.py
    # (equivalent to: uvicorn api:app --reload --port 8000)
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
