import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from typing import List, Dict
from src.retrieval.retriever import RAGRetriever
from src.safety.filters import is_safe_question, sanitize_response


class RAGGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        device: str = None,
        max_new_tokens: int = 40,
        temperature: float = 0.3,
        top_p: float = 0.9,
    ):
        print(f"[INFO] Chargement du modèle HuggingFace : {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if "t5" in model_name.lower() or "flan" in model_name.lower():
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.is_seq2seq = True
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.is_seq2seq = False

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        print(f"[INFO] Modèle prêt sur {self.device} ({'Seq2Seq' if self.is_seq2seq else 'Causal LM'})")

    def build_prompt(self, question: str, retrieved_chunks: List[Dict], system_prompt: str) -> str:
        # Nettoyage simple
        texts = [chunk["text"].strip() for chunk in retrieved_chunks]
        context = "\n".join(texts)

        prompt = (
            f"{system_prompt}\n\n"
            f"Texte :\n{context}\n\n"
            f"Question : {question}\n"
            "Réponse :"
        )
        return prompt

    def generate(self, question: str, retrieved_chunks: List[Dict], system_prompt: str = "") -> str:
        #max_tokens = 512 if self.is_seq2seq else 2048
        prompt = self.build_prompt(question, retrieved_chunks, system_prompt)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=min(2048, self.tokenizer.model_max_length),
        ).to(self.device)

        with torch.no_grad():
            if self.is_seq2seq:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=True,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()


def main():
    parser = argparse.ArgumentParser(description="Test de génération RAG sur le Code de la Route")
    parser.add_argument("--QUERY", type=str, required=True, help="Question à poser")
    parser.add_argument("--model", type=str, default="google/flan-t5-base", help="Modèle Hugging Face à utiliser")
    parser.add_argument("--index_path", type=str, default="data/index/faiss.index", help="Chemin vers l’index FAISS")
    parser.add_argument("--top_k", type=int, default=3, help="Nombre de passages à récupérer")
    args = parser.parse_args()

    # if not is_safe_question(args.QUERY):
    #     print("⚠️ Question non conforme. Veuillez reformuler.")
    #     return

    retriever = RAGRetriever(index_path=args.index_path)
    generator = RAGGenerator(model_name=args.model)

    system_prompt = (
        "Tu es un assistant expert du Code de la route français. "
        "Réponds uniquement à partir du texte fourni. "
        "Réponse courte, factuelle, 1 phrase maximum (<20 mots), en français."
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
        print(f"- num {r['num']} (score: {r['score']:.4f})")


if __name__ == "__main__":
    main()
