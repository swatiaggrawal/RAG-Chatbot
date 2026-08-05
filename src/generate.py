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
from google import genai

from retrieve import retrieve

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
load_dotenv()

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
GEMINI_MODEL = "gemini-3.6-flash"  # current stable Flash model (as of mid-2026), free-tier friendly

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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": 4096,  # generous ceiling so the visible answer can't
                                          # get starved out by internal "thinking" tokens
        },
    )

    answer_text = response.text
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
