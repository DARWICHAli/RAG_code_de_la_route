# src/retrieval/retriever.py
import faiss, pickle
from sentence_transformers import SentenceTransformer
import numpy as np

class Retriever:
    def __init__(self, index_path="data/index/faiss.index", model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = faiss.read_index(index_path)
        with open(index_path.replace('.index', '.meta.pkl'), 'rb') as f:
            self.meta = pickle.load(f)

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, ids = self.index.search(q_emb, top_k)
        results = []
        for idx, sc in zip(ids[0], scores[0]):
            m = self.meta[idx]
            results.append({"score": float(sc), "text": m["text"], "page": m["page"], "id": m["id"]})
        return results
