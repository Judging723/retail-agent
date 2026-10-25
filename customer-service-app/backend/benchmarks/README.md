# Agent evaluation

`agent_cases.json` is a fixed, manually reviewed evaluation set for the first Agent decision turn.
It contains 20 Chinese customer-service cases:

- 5 order, shipment, refund-progress, or multi-turn queries;
- 6 complete refund requests;
- 9 cases where the Agent must not call a tool because information is missing, the user is only
  asking a policy question, the capability is unsupported, or the message contains prompt injection.

The evaluator measures:

- tool-selection accuracy across all cases;
- end-to-end correct tool and exact normalized arguments on tool cases;
- exact argument accuracy after the correct tool has been selected;
- no-tool accuracy on cases that should answer or clarify without side effects.

It does not measure response wording or whether a database operation succeeds. Ownership, amount,
idempotency, and audit behavior are covered separately by integration tests.

Validate the dataset without spending API credits:

```bash
cd customer-service-app/backend
python scripts/evaluate_agent.py --validate-only
```

Run a live DeepSeek baseline after configuring `RETAIL_LLM_API_KEY`:

```bash
python scripts/evaluate_agent.py \
  --output benchmarks/results/deepseek-baseline.json
```

Use `--limit 3` for a low-cost smoke run. Do not use this fixed set to repeatedly tune the prompt;
that would contaminate the reported result. Add a separate development set before systematic prompt
optimization or fine-tuning. Generated result files under `benchmarks/results/` should record the exact
model name and evaluation time.

## Baseline

On 2026-10-23, `deepseek-v4-flash` produced the following first-turn result:

| Metric | Result |
| --- | ---: |
| Tool-selection accuracy | 0.850 (17/20) |
| Tool-case end-to-end accuracy | 0.818 (9/11) |
| Argument accuracy given correct tool | 1.000 (9/9) |
| No-tool accuracy | 0.889 (8/9) |

The three selection failures are retained in the result file: two explicit refund requests selected
the read tool instead of the refund tool, and one incomplete refund request called the read tool
instead of asking for the missing amount. A request explicitly identifying an order as belonging
to a friend is correctly labelled no-tool; database authorization is tested separately.
This is the untouched baseline, not a tuned result. See
[`results/deepseek-baseline.json`](results/deepseek-baseline.json) for per-case predictions and
configuration hashes.
