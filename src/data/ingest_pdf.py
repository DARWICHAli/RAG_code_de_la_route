# src/data/ingest_pdf.py
import os
from pathlib import Path
import argparse
import PyPDF2
from typing import List
import hashlib

def load_pdf(path: str) -> List[str]:
    text_pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            text_pages.append(p.extract_text() or "")
    return text_pages

def chunk_text(pages: List[str], chunk_size: int=1000, overlap: int=200):
    chunks = []
    for i, page in enumerate(pages):
        text = page.replace("\n", " ").strip()
        start = 0
        while start < len(text):
            end = min(start+chunk_size, len(text))
            chunk = text[start:end]
            chunks.append({"text": chunk, "page": i+1})
            start = max(end - overlap, end)
    return chunks

def save_chunks(chunks, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    import json
    outp = Path(out_dir) / "chunks.jsonl"
    with open(outp, "w", encoding="utf-8") as fw:
        for c in chunks:
            # add id
            id_ = hashlib.sha1((str(c["page"]) + c["text"][:50]).encode()).hexdigest()
            c_out = {"id": id_, "page": c["page"], "text": c["text"]}
            fw.write(json.dumps(c_out, ensure_ascii=False) + "\n")
    print("Saved", outp)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", required=True)
    p.add_argument("--out_dir", default="data/processed")
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--overlap", type=int, default=200)
    args = p.parse_args()

    pages = load_pdf(args.pdf)
    chunks = chunk_text(pages, args.chunk_size, args.overlap)
    save_chunks(chunks, args.out_dir)
