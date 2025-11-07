"""
PDF Ingestion & Chunking — Code de la Route
-------------------------------------------
- Sépare le plan et le corps du texte
- Découpe le texte en chunks avec overlap
- Sauvegarde en JSONL pour indexation FAISS
"""

import argparse
import json
from pathlib import Path
from PyPDF2 import PdfReader


def extract_pdf_chunks(pdf_path, chunk_size=500, overlap=100, plan_start=3, plan_end=6):
    """
    Extrait et découpe le PDF en chunks textuels avec métadonnées.
    """
    reader = PdfReader(pdf_path)
    chunks = []

    # 1️⃣ Extraire le plan séparément
    print(f"[INFO] Extraction du plan : pages {plan_start} à {plan_end}")
    plan_text = ""
    for i in range(plan_start - 1, plan_end):
        page_text = reader.pages[i].extract_text() or ""
        plan_text += page_text + "\n"

    chunks.append({
        "chunk_id": "plan_section",
        "page": f"{plan_start}-{plan_end}",
        "section": "Plan",
        "text": plan_text.strip()
    })

    # 2️⃣ Extraire le reste du PDF
    print(f"[INFO] Extraction du corps du texte à partir de la page {plan_end + 1}")
    for i in range(plan_end, len(reader.pages)):
        page_text = reader.pages[i].extract_text()
        if not page_text:
            continue

        words = page_text.split()
        start = 0
        chunk_idx = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "chunk_id": f"page{i + 1}_chunk{chunk_idx}",
                "page": i + 1,
                "section": "Code",
                "text": chunk_text.strip()
            })
            start += chunk_size - overlap
            chunk_idx += 1

    print(f"[INFO] {len(chunks)} chunks extraits depuis le PDF.")
    return chunks


def save_chunks(chunks, out_dir):
    """
    Sauvegarde les chunks en JSONL.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chunks.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[INFO] Fichier de sortie créé : {out_file} ({len(chunks)} chunks)")


def main():
    parser = argparse.ArgumentParser(description="Ingestion et chunking du PDF Code de la Route")
    parser.add_argument("--pdf", type=str, required=True, help="Chemin vers le fichier PDF")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Dossier de sortie")
    parser.add_argument("--chunk_size", type=int, default=500, help="Taille du chunk (en mots)")
    parser.add_argument("--overlap", type=int, default=100, help="Chevauchement entre les chunks")
    parser.add_argument("--plan_pages", nargs=2, type=int, default=[3, 6], help="Pages du plan à extraire séparément")

    args = parser.parse_args()

    chunks = extract_pdf_chunks(
        pdf_path=args.pdf,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        plan_start=args.plan_pages[0],
        plan_end=args.plan_pages[1],
    )

    save_chunks(chunks, args.out_dir)


if __name__ == "__main__":
    main()
