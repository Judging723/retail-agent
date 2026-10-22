# Retrieval evaluation

This directory contains the fixed retrieval evaluation set used to compare retrieval strategies.
The cases must not be used to tune document keywords, otherwise the reported metrics would be
contaminated.

## Experiment setup

- Date: 2026-10-22
- Corpus: 20 fictional ecommerce knowledge documents
- Evaluation set: 10 positive queries and 2 unrelated queries
- Ranking cutoff: 3
- Device: Apple M4 MPS
- Sentence Transformers: 6.2.0
- PyTorch: 2.14.1
- Embedding model: `BAAI/bge-small-zh-v1.5`
- Model revision: `7999e1d3359715c523056ef9478215996d62a620`
- Embedding dimension: 512
- Indexed text: document title and content
- Similarity: dot product over normalized embeddings, equivalent to cosine similarity

## Results

| Retriever | Hit@3 | MRR | Negative accuracy |
| --- | ---: | ---: | ---: |
| Exact keyword baseline | 0.400 | 0.400 | 1.000 |
| BGE semantic baseline | 1.000 | 0.950 | 0.000 |

The semantic model retrieved the relevant document for every positive query in the top three.
Nine relevant documents ranked first and one ranked second. However, dense retrieval returned
documents for both unrelated queries, so a later stage must calibrate a rejection threshold on a
separate development set instead of tuning against this fixed evaluation set.

These results describe only this small educational benchmark and are not production-quality claims.

## Reproduce

From the `knowledge-pipeline` directory:

```bash
python scripts/evaluate_semantic.py
```

The first run downloads the pinned public model into the local Hugging Face cache. Model weights are
not stored in this repository.
