# Model pipeline

This directory is the offline model-engineering boundary. It currently prepares reproducible supervised
fine-tuning (SFT) data and trains an interpretable clarification baseline.

## What the preparation step does

1. Reads the pinned MIT-licensed Chinese Ambiguous Reference dataset already converted by the knowledge pipeline.
2. Repairs only known formatting noise such as missing quotes, missing speaker separators, and wrapped lines.
   A source dialogue that starts with an assistant is excluded in full, because deleting that opening turn can
   turn a contextual answer such as “还没有” into a misleading standalone user request.
3. Maps original speakers to `user` and `assistant`, then creates one supervised sample for each assistant response.
4. Removes high-confidence phone numbers, email addresses, ID cards, bank cards, and labeled order IDs.
5. Deduplicates samples and assigns complete source conversations to stable 80/10/10 splits.
6. Writes JSONL data plus a checksum manifest so the result can be reproduced and audited.

The split is based on `source_id`, not on individual responses. Therefore different prefixes of the same
conversation cannot appear in both training and evaluation data.

## QLoRA preparation

`export_llamafactory_data.py` removes provenance fields from the prepared records and writes the OpenAI
message format plus `dataset_info.json` expected by LLaMA-Factory. The first training experiment is deliberately
narrow: teach `Qwen/Qwen3-4B-Instruct-2507` to ask a concise clarification question when a shopping request is
underspecified. It does not train order tools, refunds, platform policies, or general customer-service facts.

The follow-up guardrail experiment adds 40 manually curated business-boundary conversations. They are exported
as a separate dataset and repeated four times, producing 160 training rows without pretending they are unique
examples. The separate `business_boundary_eval.json` remains untouched for manual evaluation.

On the Linux training machine, run the checked-in single-GPU configuration from this directory:

```bash
PYTHONPATH=. python scripts/export_llamafactory_data.py
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train configs/qwen3_4b_qlora_sft.yaml
```

Use the tagged `v0.9.4` LlamaFactory release for this first experiment. It supports Qwen3-2507 and keeps the
training environment reproducible instead of following a moving `main` branch.

After recording the first experiment, run the one-epoch corrective experiment from the same base model:

```bash
PYTHONPATH=. python scripts/export_llamafactory_data.py
CUDA_VISIBLE_DEVICES=1 llamafactory-cli train configs/qwen3_4b_qlora_guardrails.yaml
```

This writes a new adapter to `artifacts/qlora/qwen3-4b-instruct-2507-guardrails`; it does not overwrite or
continue training the first adapter. The lower learning rate and single epoch reflect the observed validation
curve, whose best checkpoint occurred before the first epoch completed.

The configuration pins the base-model revision and uses 4-bit NF4 QLoRA, FP16 computation, sequence length
1024, batch size 1, and gradient accumulation 8. It targets the otherwise idle second RTX 2080 Ti; do not use
both GPUs until a measured single-GPU limitation justifies distributed training.

## Run

```bash
cd model-pipeline
python -m pip install -e ".[dev]"
cd ../knowledge-pipeline
python scripts/prepare_public_knowledge.py
cd ../model-pipeline
python scripts/prepare_sft_data.py
PYTHONPATH=. python scripts/export_llamafactory_data.py
python scripts/prepare_nlu_data.py
python scripts/train_nlu_baseline.py
python scripts/evaluate_entity_baseline.py
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

## Entity baseline

The entity baseline deliberately covers only the arguments used by the current order tools: `order_no`,
`amount`, and an explicitly phrased refund `reason`. It uses regular expressions and a 13-case hand-written
regression set. This is an interpretable starting point, not a general named-entity recognition model.

It is not wired into the online Agent. DeepSeek Function Calling already produces tool arguments, and Pydantic
validation remains the final authority. Adding a second online argument path would duplicate responsibility
without improving the current flow.
