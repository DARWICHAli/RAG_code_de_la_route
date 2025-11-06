"""
RAG Orchestrator — Code de la Route
-----------------------------------
Pipeline complet:
1. Retriever (FAISS + embeddings)
2. Generator (Hugging Face seq2seq)
3. Safety Layer
4. Logging (JSON)
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.retrieval.retriever import RAGRetriever
from src.generation.generate_hf import HFGenerator
from src.safety.filters import is_safe_question, sanitize_response

LOG_FILE = Path("logs/rag_queries.json")
LOG_FILE.parent.mkdir(exist_ok=True)

def log_query(query, retrieved, answer, retriever_model, generator_model, top_k):
    """Append query, context, answer and params to JSON log"""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "retrieved": [
            {"page": r["page"], "score": r["score"], "excerpt": r["text"][:250]}
            for r in retrieved
        ],
        "answer": answer,
        "retriever_model": retriever_model,
        "generator_model": generator_model,
        "top_k": top_k
    }

    if LOG_FILE.exists():
        data = json.loads(LOG_FILE.read_text())
    else:
        data = []

    data.append(entry)
    LOG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))



def run_rag_pipeline(
    query: str,
    retriever_model: str,
    generator_model: str,
    top_k: int = 3,
):
    """Run end-to-end retrieval + generation pipeline with safety and logging"""
    print("=== 🚦 RAG Code de la Route ===")

    # Safety check
    if not is_safe_question(query):
        print("❌ Question hors-sujet détectée !")
        return "Désolé, je ne peux répondre qu'aux questions sur le Code de la Route."

    # 1️⃣ Load retriever
    retriever = RAGRetriever(model_name=retriever_model)
    print(f"[INFO] Retriever loaded ({retriever_model})")

    # 2️⃣ Load generator
    generator = HFGenerator(model_name=generator_model)
    print(f"[INFO] Generator loaded ({generator_model})")

    # 3️⃣ Retrieve top-k chunks
    print(f"\n[INFO] Retrieving top {top_k} relevant chunks...")
    retrieved = retriever.retrieve(query, top_k=top_k)

    print("\n=== 🔍 Retrieved contexts ===")
    for i, r in enumerate(retrieved, 1):
        text_preview = r['text'][:250].replace("\n", " ") + "..."
        print(f"\n[{i}] Page {r['page']} | Score {r['score']:.3f}")
        print(f"Excerpt: {text_preview}")

    # 4️⃣ Generate answer
    print(f"\n[INFO] Generating answer with {generator_model} model...")
    answer = generator.generate(query, retrieved)

    # 5️⃣ Apply safety filter on output
    answer = sanitize_response(answer)

    print("\n=== 💬 Final Answer ===")
    print(answer)

    # 6️⃣ Log query
    log_query(query, retrieved, answer, retriever_model, generator_model, top_k)

    return answer



def main():
    parser = argparse.ArgumentParser(description="Run RAG chatbot for Code de la Route")
    parser.add_argument("--query", type=str, default="Quelles sont les règles concernant le permis à points ?")
    parser.add_argument("--retriever_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--generator_model", type=str, default="google/flan-t5-base")
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    run_rag_pipeline(
        query=args.query,
        retriever_model=args.retriever_model,
        generator_model=args.generator_model,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
