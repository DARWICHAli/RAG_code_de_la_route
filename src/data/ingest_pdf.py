import os
import re
import json
import hashlib
import argparse
from pathlib import Path
import PyPDF2


# === REGEX DEFINITIONS ===
SECTION_REGEX = {
    "livre": re.compile(r"\bLivre\s+[IVX0-9er]+\b", re.I),
    "titre": re.compile(r"\bTitre\s+[IVX0-9er]+\b", re.I),
    "chapitre": re.compile(r"\bChapitre\s+[IVX0-9er]+\b", re.I),
    "article": re.compile(r"\b[LRA]\.\s*\d+[-\d]*", re.I),
}

PLAN_LINE_REGEX = re.compile(
    r"^(?P<title>[\w\s\(\)\.\-:’'éèàâçîûô]+)\s+\.{5,}\s+(?P<page>\d+)$"
)


# === PDF LOADING ===
def load_pdf(pdf_path):
    """Extracts text from each page of the PDF."""
    pages = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for p in reader.pages:
            text = p.extract_text() or ""
            text = re.sub(r"\s+", " ", text.strip())
            pages.append(text)
    return pages


# === PLAN PARSER ===
def parse_plan(pages):
    """Extract structured plan lines (title, page) from plan pages."""
    plan_items = []
    for page_num, text in enumerate(pages, start=1):
        for line in text.split("  "):
            match = PLAN_LINE_REGEX.search(line.strip())
            if match:
                plan_items.append(
                    {
                        "title": match.group("title").strip(),
                        "page": int(match.group("page")),
                    }
                )
    return plan_items


# === CODE CHUNKING ===
def detect_sections(text, current_context):
    """Update context (Livre, Titre, Chapitre, Article) based on text content."""
    for key, regex in SECTION_REGEX.items():
        match = regex.search(text)
        if match:
            current_context[key] = match.group()
    return current_context


def chunk_text(pages, chunk_size=1000, overlap=200, start_page=1):
    """Split PDF text into chunks with hierarchical context."""
    chunks = []
    current_context = {}

    for i, page_text in enumerate(pages, start=start_page):
        current_context = detect_sections(page_text, current_context)
        text = page_text.replace("\n", " ").strip()

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            part = text[start:end]

            chunk_id = hashlib.sha1((str(i) + part[:60]).encode()).hexdigest()
            chunk = {
                "id": chunk_id,
                "page": i,
                "context": " | ".join(
                    [f"{k.capitalize()}: {v}" for k, v in current_context.items()]
                ),
                "text": part,
            }
            chunks.append(chunk)
            start = max(end - overlap, end)
    return chunks


# === MAIN ===
def main(pdf_path, out_dir, chunk_size=1000, overlap=200, plan_pages=(3, 6)):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # 1️⃣ Load PDF
    pages = load_pdf(pdf_path)

    # 2️⃣ Extract Plan (table of contents)
    plan_start, plan_end = plan_pages
    plan_pages_text = pages[plan_start - 1 : plan_end]
    plan = parse_plan(plan_pages_text)
    with open(Path(out_dir) / "plan.jsonl", "w", encoding="utf-8") as f:
        for item in plan:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[✓] Saved {len(plan)} plan entries → {out_dir}/plan.jsonl")

    # 3️⃣ Extract Code content (actual text)
    code_pages = pages[plan_end:]
    chunks = chunk_text(code_pages, chunk_size, overlap, start_page=plan_end + 1)
    with open(Path(out_dir) / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"[✓] Saved {len(chunks)} chunks → {out_dir}/chunks.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--out_dir", default="data/processed", help="Output directory")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--plan_pages", type=int, nargs=2, default=[3, 6])
    args = parser.parse_args()

    main(
        pdf_path=args.pdf,
        out_dir=args.out_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        plan_pages=tuple(args.plan_pages),
    )
