from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="philtalker",
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)