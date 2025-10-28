# src/data/ingest_pdf.py
import os
from pathlib import Path
import argparse
import PyPDF2
import hashlib
import json
import re

# Regex pour détecter sections
SECTION_REGEX = {
    "livre": re.compile(r"^Livre\s+\d", re.I),
    "titre": re.compile(r"^Titre\s+\d", re.I),
    "chapitre": re.compile(r"^Chapitre\s+\d", re.I),
    "article": re.compile(r"L\.\s*\d[-\d]*", re.I),
}

def load_pdf(path):
    """Extract text from PDF pages."""
    pages = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            pages.append(p.extract_text() or "")
    return pages

def parse_sections(text):
    """Detect current Livre/Titre/Chapitre/Article in a text page."""
    context = {}
    for key, regex in SECTION_REGEX.items():
        m = regex.search(text)
        if m:
            context[key] = m.group()
    return context

def chunk_text(pages, chunk_size=1000, overlap=200, context_prefix=""):
    """Split text into chunks with context."""
    chunks = []
    current_context = {}
    
    for i, page in enumerate(pages):
        text = page.replace("\n", " ").strip()
        # update context from text
        new_context = parse_sections(text)
        current_context.update(new_context)
        
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text_piece = text[start:end]
            chunk = {
                "id": hashlib.sha1((str(i)+chunk_text_piece[:50]).encode()).hexdigest(),
                "page": i+1,
                "context": context_prefix + " | ".join([f"{k}: {v}" for k,v in current_context.items()]),
                "text": chunk_text_piece
            }
            chunks.append(chunk)
            start = max(end - overlap, end)
    return chunks

def save_chunks(chunks, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = Path(out_dir) / "chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {len(chunks)} chunks to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to PDF file")
    parser.add_argument("--out_dir", default="data/processed", help="Output folder")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--plan_pages", type=int, nargs=2, default=[3,6],
                        help="Pages containing the table of contents (1-indexed, inclusive)")
    args = parser.parse_args()

    pages = load_pdf(args.pdf)

    # Séparer plan et texte réel
    plan_start, plan_end = args.plan_pages
    plan_pages = pages[plan_start-1 : plan_end]  # 1-indexed
    code_pages = pages[plan_end:]  # à partir de la page suivante

    # Chunk plan séparément (facultatif)
    plan_chunks = chunk_text(plan_pages, args.chunk_size, args.overlap, context_prefix="Table des matières | ")
    code_chunks = chunk_text(code_pages, args.chunk_size, args.overlap)

    # Combiner
    chunks = plan_chunks + code_chunks

    # Sauvegarder
    save_chunks(chunks, args.out_dir)
