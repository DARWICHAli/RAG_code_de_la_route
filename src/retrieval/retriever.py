# src/retrieval/retriever.py
import faiss
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


class RAGRetriever:
    def __init__(self, index_path="data/index/faiss_index.bin",
                 metadata_path="data/index/metadata.json",
                 model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model_name = model_name

        print(f"[INFO] Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)

        print(f"[INFO] Loading metadata from {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"[INFO] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def embed_query(self, query: str):
        embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        return embedding

    def retrieve(self, query: str, top_k=5):
        """Return top_k most similar chunks"""
        embedding = self.embed_query(query)
        scores, indices = self.index.search(embedding, top_k)

        results = []
        for idx, score in zip(indices[0], scores[0]):
            meta = self.metadata.get(str(idx)) or self.metadata.get(int(idx))
            results.append({
                "score": float(score),
                "id": meta["id"],
                "page": meta["page"],
                "context": meta["context"],
                "text": self._get_text_from_chunk(idx)
            })
        return results

    def _get_text_from_chunk(self, idx):
        # optional placeholder – could load full text if stored separately
        return f"[Chunk {idx}] text not loaded (extend retriever to fetch full text)."


if __name__ == "__main__":
    retriever = RAGRetriever()
    query = "Quelles sont les règles concernant le permis à points ?"
    results = retriever.retrieve(query, top_k=3)
    for r in results:
        print(f"\nPage {r['page']} | Score {r['score']:.3f}")
        print(f"Context: {r['context']}")
