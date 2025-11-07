"""
Indexation des chunks du Code de la Route avec FAISS + SentenceTransformers
---------------------------------------------------------------------------
- Charge les chunks depuis JSONL
- Encode les textes avec un modèle HuggingFace léger
- Construit et sauvegarde l'index FAISS
"""

import argparse
import json
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_path):
    """
    Charge les chunks à partir du fichier JSONL.
    """
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"[INFO] {len(chunks)} chunks chargés depuis {chunks_path}")
    return chunks


def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2", index_path="data/index/faiss.index"):
    """
    Encode les chunks et construit l'index FAISS.
    """
    print(f"[INFO] Chargement du modèle d'embedding : {model_name}")
    model = SentenceTransformer(model_name)

    texts = [chunk["text"] for chunk in chunks]
    print("[INFO] Encodage des textes...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=64)

    # Normalisation L2 (important pour la similarité cosinus)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Sauvegarde de l’index et des métadonnées
    index_dir = Path(index_path).parent
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_path))
    print(f"[INFO] Index FAISS sauvegardé : {index_path}")

    # Sauvegarder les métadonnées (id -> chunk)
    meta_path = index_dir / "metadata.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[INFO] Métadonnées sauvegardées : {meta_path}")
    print(f"[INFO] Dimension des embeddings : {dim}, taille de l’index : {index.ntotal}")


def main():
    parser = argparse.ArgumentParser(description="Indexation FAISS des chunks du Code de la Route")
    parser.add_argument("--chunks", type=str, required=True, help="Fichier JSONL contenant les chunks")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Modèle SentenceTransformer")
    parser.add_argument("--index_path", type=str, default="data/index/faiss.index", help="Chemin de sortie de l'index")
    args = parser.parse_args()

    chunks = load_chunks(args.chunks)
    build_faiss_index(chunks, model_name=args.model, index_path=args.index_path)


if __name__ == "__main__":
    main()
