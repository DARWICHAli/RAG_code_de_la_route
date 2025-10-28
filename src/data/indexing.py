# src/data/indexing.py
import faiss
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import argparse

def load_chunks(path):
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            out.append(json.loads(line))
    return out

def build_faiss(chunks, model_name="all-MiniLM-L6-v2", index_path="data/index/faiss.index"):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    # save metadata
    import pickle
    with open(Path(index_path).with_suffix('.meta.pkl'), 'wb') as fw:
        pickle.dump(chunks, fw)
    print("Index saved:", index_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", default="data/processed/chunks.jsonl")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--index_path", default="data/index/faiss.index")
    args = p.parse_args()
    chunks = load_chunks(args.chunks)
    build_faiss(chunks, args.model, args.index_path)
