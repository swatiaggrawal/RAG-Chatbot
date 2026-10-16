"""
Evaluation harness for the RAG pipeline.

Two independent suites, so cheap checks can run on every push and the
expensive one on demand:

  --retrieval   Did search find the right docs? For each question in
                golden.jsonl, checks whether a chunk from an expected
                source file appears in the top-k, for every retrieval
                configuration (dense / keyword / hybrid / hybrid+rerank).
                Metrics: hit@1, hit@5 and MRR@5. Follow-up questions are
                also searched raw vs rewritten, to measure what
                conversation memory is worth.

  --generation  Is the final answer any good? Runs the full pipeline,
                then an LLM judge (a different model family from the
                answer model) scores each answer for faithfulness to the
                sources, relevance and correctness against hand-written
                key facts, and checks citations and refusals. Every
                failure is classified, so it's clear whether to fix
                retrieval or generation.

Usage (from the repo root, with .env filled in):
    python evals/run_evals.py --retrieval
    python evals/run_evals.py --generation
    python evals/run_evals.py --retrieval --generation --limit 10

Results are written to evals/results/ as JSON and Markdown.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from generate import answer_question, rewrite_query  # noqa: E402
from llm import ANSWER_MODEL, FAST_MODEL, JUDGE_MODEL, complete  # noqa: E402
from retrieve import embed_query, retrieve  # noqa: E402

GOLDEN = Path(__file__).parent / "golden.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"
TOP_K = 5

RETRIEVAL_CONFIGS = {
    "dense": {"mode": "dense", "reranker": "none"},
    "keyword (BM25)": {"mode": "keyword", "reranker": "none"},
    "hybrid (RRF)": {"mode": "hybrid", "reranker": "none"},
    "hybrid + LLM rerank": {"mode": "hybrid", "reranker": "llm"},
}

JUDGE_PROMPT = """You are a strict grader for a documentation question-answering system.

You get: the user's question, the numbered sources the system retrieved, the system's answer, and reference key facts a good answer should contain.

Return JSON with exactly these fields:
- "faithfulness": 1-5. 5 = every claim in the answer is supported by the sources; 1 = mostly unsupported or invented.
- "relevance": 1-5. 5 = directly and fully addresses the question.
- "correctness": 1-5. 5 = covers all the key facts; 3 = covers some; 1 = covers none or contradicts them.
- "citations_ok": true if every [n] citation in the answer points to a source that supports the sentence it's attached to, and the answer cites at least one source when it makes factual claims. true if the answer makes no factual claims.
- "refused": true if the answer says the information isn't available in the docs / declines to answer, instead of answering.
- "reason": one short sentence explaining the lowest score.

Judge only against the sources and key facts given, not your own knowledge."""


def load_cases(limit: int | None) -> list[dict]:
    cases = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def first_hit_rank(sources: list[str], expected: list[str]) -> int | None:
    for rank, source in enumerate(sources, 1):
        if source in expected:
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict:
    n = len(ranks) or 1
    return {
        "hit@1": sum(1 for r in ranks if r == 1) / n,
        "hit@5": sum(1 for r in ranks if r is not None) / n,
        "mrr@5": sum(1 / r for r in ranks if r is not None) / n,
        "n": len(ranks),
    }


# --------------------------------------------------------------------- retrieval

def run_retrieval(cases: list[dict]) -> dict:
    cases = [c for c in cases if c["type"] != "unanswerable"]
    ranks: dict[str, list] = {name: [] for name in RETRIEVAL_CONFIGS}
    per_case = []
    followup_raw, followup_rewritten = [], []

    for i, case in enumerate(cases, 1):
        query = case["question"]
        if case["type"] == "followup":
            query = rewrite_query(case["question"], case["history"])
        vector = embed_query(query)

        row = {"id": case["id"], "query": query, "ranks": {}}
        for name, config in RETRIEVAL_CONFIGS.items():
            results = retrieve(query, top_k=TOP_K, query_vector=vector, **config)
            rank = first_hit_rank([r["source"] for r in results], case["expected_sources"])
            ranks[name].append(rank)
            row["ranks"][name] = rank
        per_case.append(row)

        if case["type"] == "followup":
            # Memory ablation: the same follow-up searched without rewriting.
            raw = retrieve(case["question"], top_k=TOP_K, mode="hybrid", reranker="llm")
            followup_raw.append(first_hit_rank([r["source"] for r in raw], case["expected_sources"]))
            followup_rewritten.append(row["ranks"]["hybrid + LLM rerank"])

        print(f"  [{i}/{len(cases)}] {case['id']}: " + ", ".join(f"{k}={v}" for k, v in row["ranks"].items()))

    return {
        "configs": {name: summarize(r) for name, r in ranks.items()},
        "followups": {
            "raw question": summarize(followup_raw),
            "rewritten with history": summarize(followup_rewritten),
        },
        "cases": per_case,
    }


# -------------------------------------------------------------------- generation

def judge(case: dict, result: dict) -> dict:
    sources = "\n\n".join(
        f"[{s['id']}] {s['source']} > {s['title']}\n{c['text']}"
        for s, c in zip(result["sources"], result["chunks"])
    ) or "(no sources retrieved)"
    history = "".join(f"{m['role']}: {m['content']}\n" for m in case.get("history", []))
    reply = complete(
        [
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": (
                f"{'Earlier conversation:' + chr(10) + history + chr(10) if history else ''}"
                f"Question: {case['question']}\n\nSources:\n{sources}\n\n"
                f"Answer:\n{result['answer']}\n\nKey facts:\n- " + "\n- ".join(case["key_facts"])
            )},
        ],
        model=JUDGE_MODEL,
        max_tokens=2048,
        temperature=0,
        json_mode=True,
    )
    return json.loads(reply)


def classify(case: dict, result: dict, grade: dict) -> str:
    """Failure taxonomy: where in the pipeline did this case go wrong?"""
    if case["type"] == "unanswerable":
        return "ok" if grade.get("refused") else "answered_unanswerable"
    if first_hit_rank([s["source"] for s in result["sources"]], case["expected_sources"]) is None:
        return "retrieval_miss"
    if grade.get("refused"):
        return "false_refusal"
    if grade.get("faithfulness", 0) < 4:
        return "unfaithful"
    if grade.get("correctness", 0) < 4:
        return "incomplete"
    if not grade.get("citations_ok"):
        return "bad_citation"
    return "ok"


def run_generation(cases: list[dict]) -> dict:
    rows = []
    for i, case in enumerate(cases, 1):
        result = answer_question(case["question"], case.get("history"))
        try:
            grade = judge(case, result)
        except Exception as e:
            grade = {"error": repr(e)}
        label = classify(case, result, grade) if "error" not in grade else "judge_error"
        rows.append({
            "id": case["id"],
            "type": case["type"],
            "label": label,
            "grade": grade,
            "search_query": result["search_query"],
            "cited": result["cited"],
            "sources": [s["source"] for s in result["sources"]],
            "timings": result["timings"],
            "answer": result["answer"],
        })
        print(f"  [{i}/{len(cases)}] {case['id']}: {label} {grade}")

    graded = [r for r in rows if r["type"] != "unanswerable" and "error" not in r["grade"]]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    avg = lambda key: sum(r["grade"].get(key, 0) for r in graded) / (len(graded) or 1)  # noqa: E731
    labels: dict[str, int] = {}
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1

    latencies = sorted(sum(r["timings"].values()) for r in rows if r["timings"])
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p90 = latencies[int(len(latencies) * 0.9) - 1] if latencies else 0

    return {
        "summary": {
            "answered questions": len(graded),
            "faithfulness (1-5)": avg("faithfulness"),
            "relevance (1-5)": avg("relevance"),
            "correctness (1-5)": avg("correctness"),
            "citation accuracy": sum(1 for r in graded if r["grade"].get("citations_ok")) / (len(graded) or 1),
            "correct refusals": sum(1 for r in unanswerable if r["label"] == "ok") / (len(unanswerable) or 1),
            "pass rate": labels.get("ok", 0) / (len(rows) or 1),
            "latency p50 (ms)": p50,
            "latency p90 (ms)": p90,
        },
        "failure_taxonomy": labels,
        "cases": rows,
    }


# ----------------------------------------------------------------------- report

def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def to_markdown(report: dict) -> str:
    lines = [
        "# Eval results",
        "",
        f"Run: {report['run_at']} · answer model `{ANSWER_MODEL}` · rerank/rewrite model "
        f"`{FAST_MODEL}` · judge `{JUDGE_MODEL}`",
        "",
    ]
    if "retrieval" in report:
        r = report["retrieval"]
        n = next(iter(r["configs"].values()))["n"]
        lines += [
            f"## Retrieval ({n} answerable questions, top-{TOP_K})",
            "",
            "| Configuration | Hit@1 | Hit@5 | MRR@5 |",
            "|---|---|---|---|",
            *[f"| {name} | {pct(m['hit@1'])} | {pct(m['hit@5'])} | {m['mrr@5']:.3f} |"
              for name, m in r["configs"].items()],
            "",
            f"**Conversation memory** ({r['followups']['raw question']['n']} follow-up questions, hybrid + rerank):",
            "",
            "| Follow-up searched as | Hit@1 | Hit@5 | MRR@5 |",
            "|---|---|---|---|",
            *[f"| {name} | {pct(m['hit@1'])} | {pct(m['hit@5'])} | {m['mrr@5']:.3f} |"
              for name, m in r["followups"].items()],
            "",
        ]
    if "generation" in report:
        g = report["generation"]
        s = g["summary"]
        lines += [
            f"## Generation ({len(g['cases'])} questions, LLM-judged)",
            "",
            "| Metric | Score |",
            "|---|---|",
            *[f"| {k} | {pct(v) if k in ('citation accuracy', 'correct refusals', 'pass rate') else (f'{v:.2f}' if isinstance(v, float) else v)} |"
              for k, v in s.items()],
            "",
            "**Failure taxonomy**",
            "",
            "| Outcome | Count |",
            "|---|---|",
            *[f"| {label} | {count} |" for label, count in sorted(g["failure_taxonomy"].items(), key=lambda kv: -kv[1])],
            "",
        ]
        failures = [c for c in g["cases"] if c["label"] != "ok"]
        if failures:
            lines += ["**Failures**", ""]
            lines += [f"- `{c['id']}` ({c['label']}): {c['grade'].get('reason', c['grade'].get('error', ''))}"
                      for c in failures]
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval", action="store_true", help="run the retrieval suite")
    parser.add_argument("--generation", action="store_true", help="run the LLM-judged generation suite")
    parser.add_argument("--limit", type=int, help="only use the first N questions")
    args = parser.parse_args()
    if not (args.retrieval or args.generation):
        parser.error("choose --retrieval and/or --generation")

    cases = load_cases(args.limit)
    report: dict = {"run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    started = time.time()

    if args.retrieval:
        print("Retrieval suite:")
        report["retrieval"] = run_retrieval(cases)
    if args.generation:
        print("Generation suite:")
        report["generation"] = run_generation(cases)

    RESULTS_DIR.mkdir(exist_ok=True)
    markdown = to_markdown(report)
    suites = "-".join(k for k in ("retrieval", "generation") if k in report)
    (RESULTS_DIR / f"latest-{suites}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"latest-{suites}.md").write_text(markdown, encoding="utf-8")

    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")

    print("\n" + markdown)
    print(f"Done in {time.time() - started:.0f}s. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
