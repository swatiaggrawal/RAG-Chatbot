"""
FastAPI wrapper around the RAG pipeline.

This file doesn't contain any RAG logic itself - it just exposes
generate.py's answer_question() as an HTTP endpoint so the frontend
(or anything else) can call it. Keeping this thin is deliberate: the
pipeline stays testable on its own via generate.py's CLI mode, and this
file's only job is the web plumbing (CORS, request/response shapes).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from generate import answer_question

app = FastAPI(title="RAG Chatbot API")

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
