import json
import os
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score
from collections import Counter
import numpy as np

from src.retrieval.retriever import RAGRetriever
from src.generation.generate_hf import HuggingFaceGenerator


def normalize_text(s):
    """Basic cleanup for text comparison."""
    import re, string
    s = s.lower().strip()
    s = re.sub(rf"[{string.punctuation}]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def exact_match(pred, ref):
    return normalize_text(pred) == normalize_text(ref)


def f1(pred, ref):
    """Token-level F1 score."""
    pred_tokens = normalize_text(pred).split()
    ref_tokens = normalize_text(ref).split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(model_name="google/flan-t5-base",
             index_path="data/index/faiss.index",
             holdout_path="data/holdout/holdout.jsonl",
             output_path="experiments/eval_results.json",
             top_k=5):

    print("[INFO] Loading retriever and generator...")
    retriever = RAGRetriever(index_path=index_path)
    generator = HuggingFaceGenerator(model_name=model_name)

    # Load holdout QA pairs
    with open(holdout_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    results = []
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    smooth_fn = SmoothingFunction().method1

    for item in tqdm(data, desc="Evaluating RAG"):
        question, ref = item["question"], item["answer"]

        # Retrieve context
        retrieved = retriever.retrieve(question, top_k=top_k)
        # Generate answer
        pred = generator.generate(question, retrieved)

        # Compute metrics
        em = exact_match(pred, ref)
        f1_val = f1(pred, ref)
        rougeL = rouge.score(ref, pred)["rougeL"].fmeasure
        bleu = sentence_bleu([ref.split()], pred.split(), smoothing_function=smooth_fn)

        results.append({
            "question": question,
            "ref": ref,
            "pred": pred,
            "exact_match": em,
            "f1": f1_val,
            "rougeL": rougeL,
            "bleu": bleu
        })

    # Aggregate results
    metrics = {
        "exact_match": np.mean([r["exact_match"] for r in results]),
        "f1": np.mean([r["f1"] for r in results]),
        "rougeL": np.mean([r["rougeL"] for r in results]),
        "bleu": np.mean([r["bleu"] for r in results]),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "details": results}, f, indent=2, ensure_ascii=False)

    print("\n=== Résultats globaux ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print(f"\n[DONE] Résultats sauvegardés dans {output_path}")


if __name__ == "__main__":
    evaluate()
