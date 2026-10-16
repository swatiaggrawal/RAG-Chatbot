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

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the context provided below.

Rules:
- If the answer isn't in the context, say "I don't have enough information to answer that" - do not make things up.
- Keep answers concise and directly grounded in the context.
- If you use information from the context, you don't need to cite chunk numbers, just answer naturally.
"""


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in chunks
    )
    return f"""Context:
{context}

Question: {question}

Answer based only on the context above:"""


def answer_question(question: str, top_k: int = 4) -> dict:
    """
    Full RAG call: retrieve relevant chunks, build a grounded prompt,
    call the LLM, and return the answer plus which sources were used
    (so you can show citations in the frontend later).
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": "No documents have been indexed yet.", "sources": []}

    prompt = build_prompt(question, chunks)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # gpt-oss models reason before answering, and reasoning tokens
        # count toward max_tokens - keep reasoning light and leave enough
        # headroom that the visible answer can't get starved out.
        reasoning_effort="low",
        max_tokens=4096,
        temperature=0.2,  # low temperature keeps answers close to the context
    )

    answer_text = response.choices[0].message.content
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
