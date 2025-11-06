"""
RAG Orchestrator — Code de la Route
-----------------------------------
This script connects:
1. Retriever (FAISS + Hugging Face embeddings)
2. Generator (Hugging Face seq2seq model)
"""

import argparse
from src.retrieval.retriever import RAGRetriever
from src.generation.generate_hf import HFGenerator


def run_rag_pipeline(
    query: str,
    retriever_model: str,
    generator_model: str,
    top_k: int = 3,
):
    """Run end-to-end retrieval + generation pipeline"""
    print("=== 🚦 RAG Code de la Route ===")

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
    print("\n[INFO] Generating answer with Hugging Face model...")
    answer = generator.generate(query, retrieved)

    print("\n=== 💬 Final Answer ===")
    print(answer)

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
