"""
Monitoring: one record per chat request, plus user feedback.

Each request produces a record with what happened (small talk, blocked
by the injection guard, or a full RAG answer), per-stage latency, tokens
used per model, whether any quota fallback kicked in, and how many
sources were retrieved and cited. Records are:

- printed as one JSON line to stdout ("[metrics] {...}"), so the hosting
  platform's log storage keeps the history (Render's free instances
  sleep and lose memory, but their logs survive), and
- kept in memory (last 2,000) to power GET /stats.

Question text is never recorded - /stats is public, and aggregate
numbers are all it needs.
"""

import json
import math
import os
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone

# Groq's free tier allows ~200K tokens per model per day; /stats shows
# usage against this so a quota problem is visible before it bites.
DAILY_TOKEN_LIMIT = int(os.environ.get("GROQ_DAILY_TOKEN_LIMIT", "200000"))
STAGES = ("guard_ms", "rewrite_ms", "search_ms", "rerank_ms", "first_token_ms", "generate_ms")


def percentile(values: list[float], p: float) -> int | None:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)])


class Metrics:
    def __init__(self, max_records: int = 2000, clock=time.time):
        self.clock = clock
        self.started_at = clock()
        self.records: deque[dict] = deque(maxlen=max_records)
        self.by_id: dict[str, dict] = {}
        self.rate_limited: deque[float] = deque(maxlen=10000)
        self.lock = threading.Lock()

    @staticmethod
    def new_request_id() -> str:
        return uuid.uuid4().hex[:12]

    def record(
        self,
        request_id: str,
        endpoint: str,
        trace: dict,
        timings: dict,
        total_ms: int,
        n_sources: int = 0,
        n_cited: int = 0,
        error: str | None = None,
    ) -> dict:
        tokens: Counter = Counter()
        for call in trace.get("llm_calls", []):
            tokens[call["model"]] += call["prompt_tokens"] + call["completion_tokens"]

        record = {
            "id": request_id,
            "ts": self.clock(),
            "endpoint": endpoint,
            "kind": trace.get("kind", "unknown"),
            "status": "error" if error else "ok",
            "error": error,
            "total_ms": total_ms,
            "timings": timings,
            "tokens": dict(tokens),
            "llm_fallback": any(c["fallback"] for c in trace.get("llm_calls", [])),
            "rerank_fallback": bool(trace.get("rerank_fallback")),
            "injection_score": trace.get("injection_score"),
            "n_sources": n_sources,
            "n_cited": n_cited,
            "feedback": None,
        }
        with self.lock:
            if len(self.records) == self.records.maxlen:
                self.by_id.pop(self.records[0]["id"], None)
            self.records.append(record)
            self.by_id[request_id] = record
        print("[metrics] " + json.dumps(record))
        return record

    def note_rate_limited(self) -> None:
        with self.lock:
            self.rate_limited.append(self.clock())

    def add_feedback(self, request_id: str, rating: str) -> bool:
        with self.lock:
            record = self.by_id.get(request_id)
            if record is None:
                return False
            record["feedback"] = rating
        print("[feedback] " + json.dumps({"id": request_id, "rating": rating}))
        return True

    def stats(self, window_s: int = 86400) -> dict:
        now = self.clock()
        with self.lock:
            recent = [r for r in self.records if now - r["ts"] < window_s]
            rate_limited = sum(1 for t in self.rate_limited if now - t < window_s)

        answered = [r for r in recent if r["kind"] == "rag" and r["status"] == "ok"]
        votes = Counter(r["feedback"] for r in recent if r["feedback"])
        tokens: Counter = Counter()
        for r in recent:
            tokens.update(r["tokens"])

        def share(items, predicate):
            return round(sum(1 for i in items if predicate(i)) / len(items), 3) if items else None

        return {
            "window_hours": window_s / 3600,
            "since": datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(timespec="seconds"),
            "requests": {
                "total": len(recent),
                "by_kind": dict(Counter(r["kind"] for r in recent)),
                "errors": sum(1 for r in recent if r["status"] == "error"),
                "rate_limited": rate_limited,
            },
            "latency_ms": {
                "total_p50": percentile([r["total_ms"] for r in answered], 50),
                "total_p90": percentile([r["total_ms"] for r in answered], 90),
                "stages_p50": {
                    stage: percentile([r["timings"][stage] for r in answered if stage in r["timings"]], 50)
                    for stage in STAGES
                },
            },
            "quality": {
                "answers_with_citations": share(answered, lambda r: r["n_cited"] > 0),
                "rerank_fallback_rate": share(answered, lambda r: r["rerank_fallback"]),
                "llm_fallbacks": sum(1 for r in recent if r["llm_fallback"]),
            },
            "feedback": {
                "up": votes.get("up", 0),
                "down": votes.get("down", 0),
                "satisfaction": round(votes["up"] / (votes["up"] + votes["down"]), 3)
                if votes["up"] + votes["down"] else None,
            },
            "tokens": [
                {"model": model, "used": used, "daily_limit": DAILY_TOKEN_LIMIT,
                 "share": round(used / DAILY_TOKEN_LIMIT, 3)}
                for model, used in tokens.most_common()
            ],
        }


metrics = Metrics()
