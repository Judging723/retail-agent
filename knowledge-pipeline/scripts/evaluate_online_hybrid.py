import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
BACKEND_ROOT = REPOSITORY_ROOT / "customer-service-app" / "backend"
EVALUATION_PATH = PROJECT_ROOT / "eval" / "retrieval_cases.json"
sys.path.insert(0, str(BACKEND_ROOT))

from app.rag import KnowledgeRetriever

from knowledge_pipeline.evaluation.metrics import (
    evaluate_retrieval,
    load_retrieval_cases,
)


def main() -> None:
    cases = load_retrieval_cases(EVALUATION_PATH)
    retriever = KnowledgeRetriever.from_milvus()

    def search(query: str, k: int) -> list[str]:
        return [result.source_id for result in retriever.search(query, limit=k)]

    metrics = evaluate_retrieval(cases, search, k=3)
    print("retriever=bm25+milvus+rrf")
    print(f"hit_at_3={metrics.hit_at_k:.3f}")
    print(f"mrr={metrics.mrr:.3f}")
    print(f"negative_accuracy={metrics.negative_accuracy:.3f}")

    for case in cases:
        results = retriever.search(case.query, limit=3)
        print(
            f"{case.case_id} | expected={list(case.relevant_source_ids)} | "
            f"retrieved={[result.source_id for result in results]}"
        )

    retriever.close()


if __name__ == "__main__":
    main()
