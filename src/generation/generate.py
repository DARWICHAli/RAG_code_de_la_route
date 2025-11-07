"""
RAG Text Generator — Code de la Route
-------------------------------------
- Combine les passages récupérés avec la question
- Construit un prompt contextuel pour génération HuggingFace
- Gère à la fois les modèles Seq2Seq (ex: FLAN-T5) et Causal LM (ex: Mistral, LLaMA)
"""

import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from typing import List, Dict
from src.retrieval.retriever import RAGRetriever  # intégré pour test complet
from src.safety.filters import is_safe_question, sanitize_response


class RAGGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        """
        Initialise le modèle et le tokenizer Hugging Face.
        """
        print(f"[INFO] Chargement du modèle HuggingFace : {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Détection automatique du type de modèle
        if "t5" in model_name.lower() or "flan" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_seq2seq = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_seq2seq = False

        # Gestion du device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"[INFO] Modèle prêt sur {self.device} ({'Seq2Seq' if self.is_seq2seq else 'Causal LM'})")

    def build_prompt(self, question: str, retrieved_chunks: List[Dict], system_prompt: str) -> str:
        """
        Construit un prompt structuré à partir du contexte et de la question.
        """
        context_blocks = []
        for r in retrieved_chunks:
            context_blocks.append(f"[Source p.{r['page']}] {r['text']}")

        context = "\n".join(context_blocks)
        prompt = (
            f"{system_prompt}\n\n"
            f"CONTEXTE :\n{context}\n\n"
            f"QUESTION : {question}\n\n"
            "RÉPONSE :"
        )
        return prompt

    def generate(self, question: str, retrieved_chunks: List[Dict], system_prompt: str = "") -> str:
        """
        Génère une réponse basée sur les passages récupérés.
        """
        prompt = self.build_prompt(question, retrieved_chunks, system_prompt)

        # Tronquer si le prompt est trop long
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="Test de génération RAG sur le Code de la Route")
    parser.add_argument("--QUERY", type=str, required=True, help="Question à poser")
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Modèle Hugging Face à utiliser")
    parser.add_argument("--index_path", type=str, default="data/index/faiss.index", help="Chemin vers l’index FAISS")
    parser.add_argument("--top_k", type=int, default=5, help="Nombre de passages à récupérer")
    args = parser.parse_args()

    if not is_safe_question(args.QUERY):
        print("⚠️ Question non conforme. Veuillez reformuler.")
        return

    retriever = RAGRetriever(index_path=args.index_path)
    generator = RAGGenerator(model_name=args.model)

    system_prompt = (
        "Tu es un assistant **français** expert du Code de la route. "
        "Réponds **uniquement en français**, de façon claire et concise. "
        "Utilise uniquement les passages fournis et cite les pages sources."
    )



    retrieved = retriever.retrieve(args.QUERY, top_k=args.top_k)
    if not retrieved:
        print("Aucune information trouvée dans la base.")
        return

    answer = generator.generate(args.QUERY, retrieved, system_prompt)
    print("\n=== Question ===")
    print(args.QUERY)
    print("\n=== Réponse ===")
    print(sanitize_response(answer))
    print("\n=== Sources ===")
    for r in retrieved:
        print(f"- Page {r['page']} (score: {r['score']:.4f})")


if __name__ == "__main__":
    main()