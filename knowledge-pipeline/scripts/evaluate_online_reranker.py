import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
BACKEND_ROOT = REPOSITORY_ROOT / "customer-service-app" / "backend"
EVALUATION_PATH = PROJECT_ROOT / "eval" / "reranker_cases.json"
sys.path.insert(0, str(BACKEND_ROOT))

from app.rag import CrossEncoderReranker, KnowledgeRetriever

from knowledge_pipeline.evaluation.metrics import evaluate_retrieval, load_retrieval_cases

CANDIDATE_LIMIT = 8
RANKING_CUTOFF = 3


def main() -> None:
    cases = load_retrieval_cases(EVALUATION_PATH)
    retriever = KnowledgeRetriever.from_milvus(
        reranker=CrossEncoderReranker(),
        rerank_candidates=CANDIDATE_LIMIT,
    )

    before_rankings = {
        case.query: [
            document.source_id
            for document in retriever.retrieve_candidates(
                case.query,
                limit=CANDIDATE_LIMIT,
            )
        ]
        for case in cases
    }
    after_rankings = {
        case.query: [
            document.source_id for document in retriever.search(case.query, limit=RANKING_CUTOFF)
        ]
        for case in cases
    }

    def search_before_reranking(query: str, k: int) -> list[str]:
        return before_rankings[query][:k]

    def search_after_reranking(query: str, k: int) -> list[str]:
        return after_rankings[query][:k]

    before = evaluate_retrieval(cases, search_before_reranking, k=RANKING_CUTOFF)
    after = evaluate_retrieval(cases, search_after_reranking, k=RANKING_CUTOFF)

    print("evaluation=hard-negative-reranking")
    print(f"candidates={CANDIDATE_LIMIT}")
    print(f"before_hit_at_3={before.hit_at_k:.3f}")
    print(f"before_mrr={before.mrr:.3f}")
    print(f"after_hit_at_3={after.hit_at_k:.3f}")
    print(f"after_mrr={after.mrr:.3f}")

    for case in cases:
        before_ids = search_before_reranking(case.query, RANKING_CUTOFF)
        after_ids = search_after_reranking(case.query, RANKING_CUTOFF)
        print(
            f"{case.case_id} | expected={list(case.relevant_source_ids)} | "
            f"before={before_ids} | after={after_ids}"
        )

    retriever.close()


if __name__ == "__main__":
    main()
