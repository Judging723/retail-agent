# Training data

`processed/` contains generated JSONL files and is intentionally ignored by Git. Recreate it from the
versioned public source with `python scripts/prepare_sft_data.py`.

`reports/sft_manifest.json` is committed. It records the input version and checksum, transformation
rules, parser recoveries, split sizes, redaction counts, and output checksums. Dialogues that start with an
assistant are excluded instead of truncating their opening context.

The NLU stage adds `reports/nlu_dataset_manifest.json` for weak-label provenance and
`reports/nlu_baseline.json` for fixed development/test metrics. Trained `.joblib` artifacts stay outside Git.

`entity_cases.json` is a small hand-written regression set for the current order-tool fields. Its report is
committed as `reports/entity_baseline.json`; a perfect score on these cases is only a regression check, not a
claim about general Chinese entity recognition.

The public dataset is suitable for learning clarification behavior. It is not a source of authoritative
refund, logistics, compensation, or legal policies.

`business_boundary_train.json` is a small, manually curated corrective set. It teaches four boundaries that
the public dialogue corpus does not cover: do not invent catalog results, do not invent policies, ask for an
order number before an order lookup, and do not claim that a business action succeeded without a tool result.
The export command repeats these 40 records four times so they remain visible beside 1,910 public training
samples. This repeat factor is an explicit sampling choice, not additional evidence or additional unique data.

`business_boundary_eval.json` contains 12 held-out prompts for manual behavioral evaluation. These prompts are
never exported into the training dataset.
