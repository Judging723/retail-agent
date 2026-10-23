# Model pipeline

This directory is the offline model-engineering boundary. It currently prepares reproducible supervised
fine-tuning (SFT) data and trains an interpretable NLU baseline for clarification detection.

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
cd ../knowledge-pipeline
python scripts/prepare_public_knowledge.py
cd ../model-pipeline
python scripts/prepare_sft_data.py
python scripts/prepare_nlu_data.py
python scripts/train_nlu_baseline.py
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

## Clarification NLU baseline

The public JSON does not directly provide a clean binary label on each user utterance. The pipeline derives
a weak label from the annotation codes on the next merged service response. It retains those original codes
in every record and documents the mapping in `data/reports/nlu_dataset_manifest.json`.

Three baselines are compared on fixed source-level splits:

- the majority class;
- an explicit keyword rule;
- character TF-IDF features with logistic regression.

Character n-grams are a useful Chinese baseline because they require neither a tokenizer nor a downloaded
neural model. The classifier is still a statistical baseline, not the final Transformer model. Its purpose is
to prove that the dataset, metrics, split, and artifact lifecycle work before expensive training is introduced.

Sentiment training is intentionally not included in this dataset: shopping-clarification strategy codes are
not sentiment labels. A sentiment stage must use a separately licensed and correctly annotated dataset.
