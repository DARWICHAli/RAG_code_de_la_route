"""
RAG Retriever — Code de la Route
--------------------------------
- Charge l'index FAISS et les métadonnées
- Récupère les passages les plus pertinents pour une question donnée
- Retourne le texte et les métadonnées (page, section, score)
"""

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path


class RAGRetriever:
    def __init__(self, index_path="data/index/faiss.index", model_name="all-MiniLM-L6-v2", top_k=5):
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.parent / "metadata.jsonl"
        self.model_name = model_name
        self.top_k = top_k

        # Charger FAISS
        print(f"[INFO] Chargement de l'index FAISS depuis {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))

        # Charger les métadonnées
        print(f"[INFO] Chargement des métadonnées depuis {self.meta_path}")
        self.metadata = self._load_metadata()

        # Charger le modèle d’embedding
        print(f"[INFO] Chargement du modèle d'embedding : {model_name}")
        self.model = SentenceTransformer(model_name)

    def _load_metadata(self):
        metadata = []
        with open(self.meta_path, "r", encoding="utf-8") as f:
            for line in f:
                metadata.append(json.loads(line))
        return metadata

    def retrieve(self, query, top_k=None):
        """
        Recherche les passages les plus similaires à la requête.
        """
        top_k = top_k or self.top_k

        query_emb = self.model.encode([query], convert_to_numpy=True)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        scores, indices = self.index.search(query_emb, top_k)
        results = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({
                "page": meta.get("page"),
                "section": meta.get("section"),
                "chunk_id": meta.get("chunk_id"),
                "text": meta.get("text")[:500] + "..." if len(meta.get("text", "")) > 500 else meta.get("text"),
                "score": float(score)
            })

        print(f"[INFO] {len(results)} passages récupérés pour la requête : '{query[:50]}...'")
        for r in results:
            print(f"  - p.{r['page']} ({r['section']}) [score={r['score']:.3f}]")

        return results
