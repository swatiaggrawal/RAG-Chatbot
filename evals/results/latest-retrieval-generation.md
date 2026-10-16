# Eval results

Run: 2026-10-16 11:48 UTC · answer model `openai/gpt-oss-120b` · rerank/rewrite model `openai/gpt-oss-20b` · judge `qwen/qwen3.8-27b`

> **Warning:** 2 of 65 rerank calls failed and fell back to hybrid order, so rerank numbers understate it. Re-run when rate limits reset.

## Retrieval (52 answerable questions, top-5)

| Configuration | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|
| dense | 90% | 100% | 0.943 |
| keyword (BM25) | 79% | 98% | 0.864 |
| hybrid (RRF) | 90% | 100% | 0.952 |
| hybrid + LLM rerank | 96% | 100% | 0.981 |

**By question type** (MRR@5 - higher means the right doc ranks nearer the top):

| Question type | n | dense | keyword (BM25) | hybrid (RRF) | hybrid + LLM rerank |
|---|---|---|---|---|---|
| standard | 34 | 0.966 | 0.848 | 0.956 | 0.971 |
| followup | 8 | 0.900 | 0.938 | 0.938 | 1.000 |
| paraphrase | 5 | 0.800 | 0.717 | 0.900 | 1.000 |
| identifier | 5 | 1.000 | 1.000 | 1.000 | 1.000 |

**Conversation memory** (8 follow-up questions, hybrid + rerank):

| Follow-up searched as | Hit@1 | Hit@5 | MRR@5 |
|---|---|---|---|
| raw question | 88% | 100% | 0.938 |
| rewritten with history | 100% | 100% | 1.000 |

## Generation (57 questions, LLM-judged)

| Metric | Score |
|---|---|
| answered questions | 52 |
| faithfulness (1-5) | 4.96 |
| relevance (1-5) | 4.96 |
| correctness (1-5) | 4.85 |
| citation accuracy | 94% |
| correct refusals | 100% |
| pass rate | 89% |
| latency p50 (ms) | 2104 |
| latency p90 (ms) | 3240 |

**Failure taxonomy**

| Outcome | Count |
|---|---|
| ok | 51 |
| incomplete | 3 |
| bad_citation | 3 |

**Failures**

- `debugging` (incomplete): The answer omits the critical code setup required to run the app in the debugger (importing uvicorn and using the `if __name__ == "__main__":` guard), which are key facts listed in the reference.
- `numeric-validation` (incomplete): The answer fails to mention the use of `Annotated`, which is a key fact required by the reference.
- `examples` (incomplete): The answer misses the key methods `model_config` and `Field(examples=...)` listed in the key facts, focusing only on `Body` arguments.
- `sse` (bad_citation): Citations [4] and [1] do not support the specific claims about importing EventSourceResponse or the exact line ranges cited, as those details are found in sources [3] and [1] respectively but the citation keys are mismatched or incorrect.
- `fu-required-query` (bad_citation): The answer suggests using `Query(...)` for required parameters, but the sources explicitly state that you can simply not declare a default value (e.g., `q: str`) and that `Query` is not needed if no other validations are required.
- `id-last-event-id` (bad_citation): The citation [1] does not contain the specific details about Last-Event-ID, which are found in source [2].
