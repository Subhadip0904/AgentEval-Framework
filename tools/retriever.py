"""
FAISS Retriever Module - vector search over specification documents
"""
import os
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class FAISSRetriever:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", index_path: str = "data/faiss_index"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index_path = index_path
        self.index = None
        self.documents = []

    def build_index(self, documents: List[Dict[str, str]]) -> None:
        """Build FAISS index from documents"""
        self.documents = documents
        texts = [doc["text"] for doc in documents]

        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        print(f"Index saved to {self.index_path}")

    def load_index(self) -> bool:
        """Load existing FAISS index"""
        if not os.path.exists(self.index_path):
            return False
        try:
            self.index = faiss.read_index(self.index_path)
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """Retrieve top-k documents similar to query"""
        if self.index is None:
            raise ValueError("Index not initialized. Call build_index() first.")

        query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx in indices[0]:
            if idx >= 0:
                results.append(self.documents[idx])

        return results
