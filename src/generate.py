"""
STAGE 4: GENERATE
--------------------
Job: take the user's question + the retrieved chunks, build a prompt
that grounds the LLM in that context, and get an answer back.

Key design choice: the prompt explicitly tells the model to only use
the provided context and to say so if the answer isn't in there. This
is what makes it "RAG" instead of just "an LLM that happens to have
some text pasted above it" - the instruction to stay grounded is what
reduces (not eliminates) hallucination.
"""

import os
import re
import sys
from dotenv import load_dotenv
from groq import Groq

from retrieve import retrieve

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# Groq's free tier covers this model; override via GROQ_MODEL in .env
# (e.g. "openai/gpt-oss-20b" for higher free-tier rate limits). Groq
# retires models regularly - see https://console.groq.com/docs/deprecations
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are FastAPI Docs Copilot, a friendly, upbeat assistant that helps developers understand FastAPI. You answer using ONLY the documentation context provided with each question.

Rules:
- Stay grounded: every fact must come from the context. If the context doesn't cover the question, say so kindly (e.g. "I couldn't find that in the FastAPI docs I have"), then suggest a related topic you can help with. Never make things up.
- Be warm and conversational, like a helpful teammate, but get to the point - no filler.
- Format with Markdown: short paragraphs, bullet points for lists, **bold** for key terms, and fenced ```python code blocks when showing code from the context.
- Don't mention "the context" or chunk numbers - just answer naturally.
"""

SMALL_TALK_PROMPT = """You are FastAPI Docs Copilot, a friendly, upbeat assistant that answers questions about FastAPI using its official documentation.

The user is making small talk (a greeting, thanks, goodbye, or asking what you can do). Reply warmly in 1-3 short sentences. When it fits, invite them to ask about FastAPI and suggest one or two example topics such as path parameters, request bodies, dependencies, middleware, CORS, background tasks, or testing. Use Markdown sparingly."""

# Greetings / thanks / "what can you do" don't need the docs - answering
# them through retrieval made the bot reply "I don't have enough
# information" to "hi". These skip retrieval and get a friendly reply.
SMALL_TALK_RE = re.compile(
    r"^\s*(hi+|hey+|hello+|hiya|yo|sup|howdy|greetings|good (morning|afternoon|evening)"
    r"|thanks?( you)?|thank u|thx|ty|cheers|bye|goodbye|see (you|ya)|ok(ay)?|cool|nice|great|awesome"
    r"|how are you( doing)?|who are you|what are you|what can you do|help)"
    r"[\s!.?,:)]*(there|bot|copilot|buddy|friend)?[\s!.?,:)]*$",
    re.IGNORECASE,
)


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""


def _complete(system: str, user: str) -> str:
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # gpt-oss models reason before answering, and reasoning tokens
        # count toward max_tokens - keep reasoning light and leave enough
        # headroom that the visible answer can't get starved out.
        reasoning_effort="low",
        max_tokens=4096,
        temperature=0.3,
    )
    return response.choices[0].message.content


def answer_question(question: str, top_k: int = 4) -> dict:
    """
    Full RAG call: retrieve relevant chunks, build a grounded prompt,
    call the LLM, and return the answer plus which sources were used
    (so you can show citations in the frontend later).
    """
    if SMALL_TALK_RE.match(question):
        return {"answer": _complete(SMALL_TALK_PROMPT, question), "sources": []}

    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": "No documents have been indexed yet.", "sources": []}

    prompt = build_prompt(question, chunks)

    answer_text = _complete(SYSTEM_PROMPT, prompt)
    sources = list({c["source"] for c in chunks})  # dedupe

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    # End-to-end manual test: ask a real question, see the real final answer.
    # This is your "does the whole v1 pipeline actually work" smoke test.
    print("RAG test mode. Type a question (or 'quit' to exit).\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() in ("quit", "exit"):
            break

        result = answer_question(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")
        print("\n" + "-" * 50 + "\n")
