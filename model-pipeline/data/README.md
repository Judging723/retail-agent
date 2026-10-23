# Training data

`processed/` contains generated JSONL files and is intentionally ignored by Git. Recreate it from the
versioned public source with `python scripts/prepare_sft_data.py`.

`reports/sft_manifest.json` is committed. It records the input version and checksum, transformation
rules, parser recoveries, split sizes, redaction counts, and output checksums.

The NLU stage adds `reports/nlu_dataset_manifest.json` for weak-label provenance and
`reports/nlu_baseline.json` for fixed development/test metrics. Trained `.joblib` artifacts stay outside Git.

The sentiment stage adds `reports/sentiment_dataset_manifest.json` and
`reports/sentiment_baseline.json`. Its downloaded CC BY 4.0 raw files, processed JSONL files, and trained
artifact stay outside Git; the committed reports retain the exact revision, checksums, split policy, label
provenance, and metrics.

`entity_cases.json` is a small hand-written regression set for the current order-tool fields. Its report is
committed as `reports/entity_baseline.json`; a perfect score on these cases is only a regression check, not a
claim about general Chinese entity recognition.

The public dataset is suitable for learning clarification behavior. It is not a source of authoritative
refund, logistics, compensation, or legal policies.
