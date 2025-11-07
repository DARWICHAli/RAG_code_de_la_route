"""
Évaluation du système RAG — Code de la route
--------------------------------------------
- Évalue la qualité de génération (Exact Match, F1, ROUGE, BLEU)
- Compatible avec RAGGenerator et RAGRetriever
- Supporte holdout JSONL simple (question + expected_answer)
"""

import json
import os
from tqdm import tqdm
from typing import List, Dict
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.retrieval.retriever import RAGRetriever
from src.generation.generate import RAGGenerator

# =====================
# Helpers
# =====================

def load_holdout(path: str) -> List[Dict]:
    """Charge le jeu de test JSONL (question, expected_answer)."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if all(k in data for k in ["question", "expected_answer"]):
                samples.append(data)
    return samples

def compute_metrics(pred: str, ref: str) -> Dict[str, float]:
    """Calcule Exact Match, F1, ROUGE-L et BLEU."""
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()

    exact = 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0

    common = set(pred_tokens) & set(ref_tokens)
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = rouge.score(ref, pred)["rougeL"].fmeasure

    bleu = sentence_bleu(
        [ref_tokens],
        pred_tokens,
        smoothing_function=SmoothingFunction().method1
    )

    return {
        "exact_match": exact,
        "f1": f1,
        "rougeL": rouge_l,
        "bleu": bleu
    }

# =====================
# Main evaluation
# =====================

def evaluate(
    holdout_path: str,
    index_path: str,
    model_name: str,
    output_path: str = "experiments/results_eval.json",
    top_k: int = 5
):
    print("[INFO] Chargement du retriever et du générateur...")
    retriever = RAGRetriever(index_path=index_path)
    generator = RAGGenerator(model_name=model_name)

    samples = load_holdout(holdout_path)
    print(f"[INFO] {len(samples)} questions de test chargées.\n")

    results = []
    agg_scores = {"exact_match": 0, "f1": 0, "rougeL": 0, "bleu": 0}

    for s in tqdm(samples, desc="Évaluation RAG"):
        question = s["question"]
        expected = s["expected_answer"]

        retrieved = retriever.retrieve(question, top_k=top_k)
        answer = generator.generate(
            question,
            retrieved,
            system_prompt="Tu es un assistant du Code de la route français."
        )

        metrics = compute_metrics(answer, expected)
        for k in agg_scores:
            agg_scores[k] += metrics[k]

        results.append({
            "question": question,
            "expected": expected,
            "answer": answer,
            "retrieved_pages": [r["page"] for r in retrieved],
            "metrics": metrics
        })

    # Moyenne globale
    for k in agg_scores:
        agg_scores[k] /= len(samples)

    print("\n=== Résultats globaux ===")
    for k, v in agg_scores.items():
        print(f"{k}: {v:.3f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "global_scores": agg_scores,
            "details": results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] Évaluation terminée — résultats sauvegardés dans {output_path}")


if __name__ == "__main__":
    evaluate(
        holdout_path="data/eval/holdout.jsonl",
        index_path="data/index/faiss.index",
        model_name="plguillou/t5-base-fr-sum-cnndm"
    )
