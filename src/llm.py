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
import time

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

# When a model's *daily* quota runs out, retrying is pointless for hours,
# so switch to a backup model instead - each Groq model has its own quota.
# Without this, the reranker silently degraded to plain hybrid order for
# the rest of the day once gpt-oss-20b hit its 200K tokens/day limit.
FALLBACKS = {
    "openai/gpt-oss-120b": "openai/gpt-oss-20b",
    "openai/gpt-oss-20b": "qwen/qwen3.8-27b",
    "qwen/qwen3.8-27b": "openai/gpt-oss-20b",
}
EXHAUSTED_COOLDOWN_S = 15 * 60
_exhausted_until: dict[str, float] = {}


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


def _is_daily_limit(error: Exception) -> bool:
    return isinstance(error, groq.RateLimitError) and any(
        marker in str(error) for marker in ("tokens per day", "requests per day", "(TPD)", "(RPD)")
    )


def _with_fallback(call, model: str):
    """Runs call(model), moving down FALLBACKS while models are out of daily quota."""
    tried = []
    while model and model not in tried:
        tried.append(model)
        if _exhausted_until.get(model, 0) > time.time():
            model = FALLBACKS.get(model)
            continue
        try:
            return call(model)
        except groq.RateLimitError as e:
            if not _is_daily_limit(e):
                raise
            _exhausted_until[model] = time.time() + EXHAUSTED_COOLDOWN_S
            print(f"[llm] {model} is out of daily quota, falling back to {FALLBACKS.get(model)}")
            model = FALLBACKS.get(model)
    raise RuntimeError(f"All models are out of daily quota: {tried}")


def complete(
    messages: list[dict],
    model: str = ANSWER_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    extra = {"response_format": {"type": "json_object"}} if json_mode else {}
    response = _with_fallback(
        lambda m: _create(messages, _params(m, max_tokens, temperature), **extra), model
    )
    return THINK_RE.sub("", response.choices[0].message.content or "").strip()


def stream(
    messages: list[dict],
    model: str = ANSWER_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.3,
):
    """Yields the answer text as it's generated (reasoning tokens excluded)."""
    chunks = _with_fallback(
        lambda m: _create(messages, _params(m, max_tokens, temperature), stream=True), model
    )
    for chunk in chunks:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta
