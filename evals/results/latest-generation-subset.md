# Eval results

Run: 2026-10-16 12:16 UTC · answer model `openai/gpt-oss-120b` · rerank/rewrite model `openai/gpt-oss-20b` · judge `qwen/qwen3.8-27b`

## Generation (6 questions, LLM-judged)

| Metric | Score |
|---|---|
| answered questions | 6 |
| faithfulness (1-5) | 4.83 |
| relevance (1-5) | 4.33 |
| correctness (1-5) | 3.67 |
| citation accuracy | 100% |
| correct refusals | 0% |
| pass rate | 50% |
| latency p50 (ms) | 17727 |
| latency p90 (ms) | 17897 |

**Failure taxonomy**

| Outcome | Count |
|---|---|
| ok | 3 |
| incomplete | 2 |
| false_refusal | 1 |

**Failures**

- `numeric-validation` (incomplete): The answer uses the deprecated `Path(...)` syntax instead of the recommended `Annotated` syntax specified in the key facts.
- `examples` (incomplete): The answer misses the key methods of using `model_config` in Pydantic models and `Field(examples=...)`, covering only `Body(examples=...)`.
- `id-last-event-id` (false_refusal): The answer refuses to answer despite Source [1] explicitly stating that Last-Event-ID is used to resume the stream from where the client left off.
