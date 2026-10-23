# Training data

`processed/` contains generated JSONL files and is intentionally ignored by Git. Recreate it from the
versioned public source with `python scripts/prepare_sft_data.py`.

`reports/sft_manifest.json` is committed. It records the input version and checksum, transformation
rules, parser recoveries, split sizes, redaction counts, and output checksums.

The NLU stage adds `reports/nlu_dataset_manifest.json` for weak-label provenance and
`reports/nlu_baseline.json` for fixed development/test metrics. Trained `.joblib` artifacts stay outside Git.

The public dataset is suitable for learning clarification behavior. It is not a source of authoritative
refund, logistics, compensation, or legal policies.
