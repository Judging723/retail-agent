# Retrieval evaluation

This directory contains the fixed retrieval evaluation set used to compare retrieval strategies.
The cases must not be used to tune document keywords, otherwise the reported metrics would be
contaminated.

## Experiment setup

- Date: 2026-10-22
- Corpus: 1,073 deduplicated Chinese Ambiguous Reference shopping dialogues
- Evaluation set: 10 positive queries and 2 unrelated queries
- Ranking cutoff: 3
- Device: Apple M4 CPU
- Sentence Transformers: 6.2.0
- PyTorch: 2.14.1
- Embedding model: `BAAI/bge-small-zh-v1.5`
- Model revision: `7999e1d3359715c523056ef9478215996d62a620`
- Embedding dimension: 512
- Indexed text: document title and content
- Similarity: dot product over normalized embeddings, equivalent to cosine similarity
- Relevance threshold: 0.52, selected on the separate `relevance_development.json` set
- Lexical channel: BM25 over product names and keywords, gated by complete product phrases
- Fusion: Reciprocal Rank Fusion with `k=60`

## Results

| Retriever | Hit@3 | MRR | Negative accuracy |
| --- | ---: | ---: | ---: |
| Exact keyword baseline | 1.000 | 1.000 | 1.000 |
| BGE semantic baseline + threshold | 1.000 | 0.950 | 1.000 |
| Online BM25 + Milvus + RRF | 1.000 | 1.000 | 1.000 |

The semantic model retrieved the relevant document for every positive query in the top three.
Nine relevant documents ranked first and one ranked second. Raw dense retrieval always returns a
nearest neighbor, including for unrelated questions. A separate development set selected 0.5172 as
the midpoint between its highest negative and lowest positive score; the application uses the
rounded threshold 0.52. The fixed test set was not used to choose that value.

The hybrid retriever moved the water-purifier case from rank two to rank one while preserving both
negative rejections. An initial BM25 version matched the word “装饰” in a programming question to
the product “黑板装饰墙贴”; the final lexical gate requires a complete product phrase and the fixed
evaluation keeps this failure as a regression case.

The keyword baseline is intentionally easy because each positive query names a unique product.
These results describe only this small educational benchmark and are not production-quality claims.

## Reproduce

From the `knowledge-pipeline` directory:

```bash
python scripts/evaluate_semantic.py
python scripts/calibrate_milvus_threshold.py
python scripts/evaluate_online_hybrid.py
```

The first run downloads the pinned public model into the local Hugging Face cache. Model weights are
not stored in this repository.
