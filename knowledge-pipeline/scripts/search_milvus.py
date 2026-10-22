import argparse
import os
from pathlib import Path

from pymilvus import MilvusClient

from knowledge_pipeline.retrieval.embeddings import SentenceTransformerEmbedder
from knowledge_pipeline.retrieval.milvus_store import MilvusVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "vector_store" / "retail_agent.db"
COLLECTION_NAME = "retail_knowledge"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local ecommerce knowledge collection")
    parser.add_argument("query", help="Chinese customer-service question")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    embedder = SentenceTransformerEmbedder()
    query_embedding = embedder.encode_queries([args.query])[0]
    client = MilvusClient(uri=os.getenv("MILVUS_URI", str(DEFAULT_DATABASE_PATH)))
    store = MilvusVectorStore(client, collection_name=COLLECTION_NAME)
    store.load()

    for rank, result in enumerate(store.search(query_embedding, limit=args.limit), start=1):
        print(
            f"{rank}. score={result.score:.4f} source={result.source_id} "
            f"title={result.title} text={result.text}"
        )
    client.close()


if __name__ == "__main__":
    main()
