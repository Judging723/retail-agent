import os
from pathlib import Path

from pymilvus import MilvusClient

from knowledge_pipeline.ingestion.chunker import chunk_document
from knowledge_pipeline.ingestion.loader import load_documents
from knowledge_pipeline.retrieval.embeddings import SentenceTransformerEmbedder
from knowledge_pipeline.retrieval.milvus_store import MilvusVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
KNOWLEDGE_PATH = REPOSITORY_ROOT / "customer-service-app" / "backend" / "data" / "knowledge.json"
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "vector_store" / "retail_agent.db"
COLLECTION_NAME = "retail_knowledge"


def main() -> None:
    milvus_uri = os.getenv("MILVUS_URI", str(DEFAULT_DATABASE_PATH))
    if milvus_uri == str(DEFAULT_DATABASE_PATH):
        DEFAULT_DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    documents = load_documents(KNOWLEDGE_PATH)
    chunks = [chunk for document in documents for chunk in chunk_document(document)]

    embedder = SentenceTransformerEmbedder()
    embeddings = embedder.encode_documents([f"{chunk.title}：{chunk.text}" for chunk in chunks])
    if not embeddings:
        raise RuntimeError("cannot build a Milvus collection without knowledge chunks")

    client = MilvusClient(uri=milvus_uri)
    store = MilvusVectorStore(client, collection_name=COLLECTION_NAME)
    store.replace_collection(vector_size=len(embeddings[0]))
    store.upsert(chunks, embeddings)

    print(f"uri={milvus_uri}")
    print(f"collection={COLLECTION_NAME}")
    print(f"documents={len(documents)}")
    print(f"chunks={len(chunks)}")
    print(f"vector_size={len(embeddings[0])}")
    client.close()


if __name__ == "__main__":
    main()
