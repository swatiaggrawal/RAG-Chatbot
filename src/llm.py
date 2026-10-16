"""
Shared Groq client and model choices.

Groq's free tier rate-limits each model separately (8K tokens/minute,
200K tokens/day at the time of writing), so the pipeline spreads its
work across three models:

- ANSWER_MODEL writes the user-facing answer (largest model).
- FAST_MODEL does the cheap helper jobs: query rewriting and reranking.
- JUDGE_MODEL grades answers in the eval harness. A different model
  family from the answer model also avoids a judge favouring its own
  writing style.

All three are overridable via env vars, since Groq retires models
regularly - see https://console.groq.com/docs/deprecations
"""

import os
import re

import groq
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# The SDK retries 429s itself, honouring Groq's retry-after header, which
# smooths over the per-minute token limits during eval runs.
client = Groq(api_key=os.environ.get("GROQ_API_KEY"), max_retries=6)

ANSWER_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
FAST_MODEL = os.environ.get("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.environ.get("GROQ_JUDGE_MODEL", "qwen/qwen3.8-27b")

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _params(model: str, max_tokens: int, temperature: float) -> dict:
    params = {"model": model, "max_tokens": max_tokens, "temperature": temperature}
    # Reasoning models spend tokens "thinking" before answering, and those
    # count toward max_tokens - keep it light so answers aren't starved.
    if model.startswith("openai/gpt-oss"):
        params["reasoning_effort"] = "low"
    elif model.startswith("qwen/"):
        params["reasoning_format"] = "hidden"
    return params


def _create(messages: list[dict], params: dict, **extra):
    try:
        return client.chat.completions.create(messages=messages, **params, **extra)
    except groq.BadRequestError as e:
        # If a model doesn't accept a reasoning parameter, retry without it
        # rather than failing the whole request.
        if "reasoning" not in str(e):
            raise
        params = {k: v for k, v in params.items() if not k.startswith("reasoning")}
        return client.chat.completions.create(messages=messages, **params, **extra)


def complete(
    messages: list[dict],
    model: str = ANSWER_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _create(messages, _params(model, max_tokens, temperature), **extra)
    return THINK_RE.sub("", response.choices[0].message.content or "").strip()


def stream(
    messages: list[dict],
    model: str = ANSWER_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.3,
):
    """Yields the answer text as it's generated (reasoning tokens excluded)."""
    for chunk in _create(messages, _params(model, max_tokens, temperature), stream=True):
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
