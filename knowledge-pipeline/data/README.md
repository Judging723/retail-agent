# Dataset workspace

This directory separates versioned provenance from downloaded data:

- `dataset_registry.json` is committed and records source, license, access, and intended use.
- `raw/` is local-only source data and is ignored by Git.
- `processed/` is local-only generated training or evaluation data and is ignored by Git.
- `vector_store/` contains the local Milvus Lite database and is ignored by Git.

`scripts/prepare_public_knowledge.py` converts the approved Chinese Ambiguous Reference source
into the versioned online retrieval artifact at
`customer-service-app/backend/data/knowledge.json`. That artifact is committed together with
its attribution and license so the demo remains reproducible.

Adding a URL does not authorize use. A dataset may enter an experiment only when its
`review_status` permits it and the experiment follows the recorded license requirements.

Public dialogue datasets may be retrieved as behavioral examples for clarification. They must
not be treated as authoritative refund, logistics, compensation, or platform-policy knowledge.
