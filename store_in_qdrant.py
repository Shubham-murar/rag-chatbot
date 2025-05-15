# store_in_qdrant.py
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings

class KnowledgeBaseManager:
    def __init__(self, collection_name: str = "ai_clone", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self._ensure_collection_exists()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def _ensure_collection_exists(self):
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )

    def store_text_file(self, file_path: str, chunks: list[Document]):
        from qdrant_client.http.models import PointStruct
        points = []
        for i, doc in enumerate(chunks):
            emb = self.embeddings.embed_query(doc.page_content)
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={
                    "page_content": doc.page_content,
                    "source": file_path,
                    "chunk_index": i
                }
            ))
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return points