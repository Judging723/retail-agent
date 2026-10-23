# Model pipeline

This directory is the offline model-engineering boundary. The first deliverable prepares reproducible
supervised fine-tuning (SFT) data; it does not train a model yet.

## What the preparation step does

1. Reads the pinned MIT-licensed Chinese Ambiguous Reference dataset already converted by the knowledge pipeline.
2. Repairs only known formatting noise such as missing quotes, missing speaker separators, and wrapped lines.
3. Maps original speakers to `user` and `assistant`, then creates one supervised sample for each assistant response.
4. Removes high-confidence phone numbers, email addresses, ID cards, bank cards, and labeled order IDs.
5. Deduplicates samples and assigns complete source conversations to stable 80/10/10 splits.
6. Writes JSONL data plus a checksum manifest so the result can be reproduced and audited.

The split is based on `source_id`, not on individual responses. Therefore different prefixes of the same
conversation cannot appear in both training and evaluation data.

## Run

```bash
cd model-pipeline
python -m pip install -e ".[dev]"
python scripts/prepare_sft_data.py
pytest -v
ruff check .
```

Generated files use a common chat format:

```json
{"id":"...","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}],"metadata":{"source_id":"...","source_url":"...","source_revision":"...","license":"MIT"}}
```

Regex pseudonymization is deliberately described as a baseline, not anonymization. It reliably catches
structured identifiers but cannot prove that free-form names and addresses have been removed. Before using
private business conversations, add entity recognition, manual sampling, access control, and legal review.
