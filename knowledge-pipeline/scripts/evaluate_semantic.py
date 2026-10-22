from pathlib import Path
from time import perf_counter

from knowledge_pipeline.embeddings import SentenceTransformerEmbedder
from knowledge_pipeline.evaluation import evaluate_retrieval, load_retrieval_cases
from knowledge_pipeline.loader import load_documents
from knowledge_pipeline.semantic import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
KNOWLEDGE_PATH = REPOSITORY_ROOT / "customer-service-app" / "backend" / "data" / "knowledge.json"
EVALUATION_PATH = PROJECT_ROOT / "evaluation" / "retrieval_cases.json"


def main() -> None:
    documents = load_documents(KNOWLEDGE_PATH)
    cases = load_retrieval_cases(EVALUATION_PATH)

    started_at = perf_counter()
    embedder = SentenceTransformerEmbedder()
    retriever = SemanticRetriever(documents, embedder)
    load_seconds = perf_counter() - started_at

    def search(query: str, k: int) -> list[str]:
        return [result.source_id for result in retriever.search(query, limit=k)]

    evaluation_started_at = perf_counter()
    metrics = evaluate_retrieval(cases, search, k=3)
    evaluation_seconds = perf_counter() - evaluation_started_at

    print(f"model={embedder.model_name}")
    print(f"revision={embedder.revision}")
    print(f"device={embedder.device}")
    print(f"model_and_index_load_seconds={load_seconds:.3f}")
    print(f"evaluation_seconds={evaluation_seconds:.3f}")
    print(f"hit_at_3={metrics.hit_at_k:.3f}")
    print(f"mrr={metrics.mrr:.3f}")
    print(f"negative_accuracy={metrics.negative_accuracy:.3f}")

    for case in cases:
        results = retriever.search(case.query, limit=3)
        formatted_results = ", ".join(
            f"{result.source_id}:{result.score:.4f}" for result in results
        )
        print(
            f"{case.case_id} | expected={list(case.relevant_source_ids)} | "
            f"retrieved=[{formatted_results}]"
        )


if __name__ == "__main__":
    main()
