from pathlib import Path

from pymilvus import MilvusClient

from knowledge_pipeline.evaluation.metrics import load_retrieval_cases
from knowledge_pipeline.evaluation.thresholds import select_relevance_threshold
from knowledge_pipeline.retrieval.embeddings import SentenceTransformerEmbedder
from knowledge_pipeline.retrieval.milvus_store import MilvusVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_PATH = PROJECT_ROOT / "eval" / "relevance_development.json"
DATABASE_PATH = PROJECT_ROOT / "data" / "vector_store" / "retail_agent.db"
COLLECTION_NAME = "retail_knowledge"


def main() -> None:
    cases = load_retrieval_cases(DEVELOPMENT_PATH)
    embedder = SentenceTransformerEmbedder()
    embeddings = embedder.encode_queries([case.query for case in cases])

    client = MilvusClient(uri=str(DATABASE_PATH))
    store = MilvusVectorStore(client, collection_name=COLLECTION_NAME)
    store.load()

    labeled_scores: list[tuple[float, bool]] = []
    for case, embedding in zip(cases, embeddings, strict=True):
        results = store.search(embedding, limit=1)
        top_score = results[0].score if results else 0.0
        is_positive = bool(case.relevant_source_ids)
        labeled_scores.append((top_score, is_positive))
        print(
            f"{case.case_id} | label={'positive' if is_positive else 'negative'} "
            f"| top_score={top_score:.4f}"
        )

    selection = select_relevance_threshold(labeled_scores)
    print(f"recommended_threshold={selection.threshold:.4f}")
    print(f"balanced_accuracy={selection.balanced_accuracy:.3f}")
    print(f"positive_recall={selection.positive_recall:.3f}")
    print(f"negative_rejection={selection.negative_rejection:.3f}")
    client.close()


if __name__ == "__main__":
    main()
