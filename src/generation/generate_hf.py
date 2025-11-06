# src/generation/generate_hf.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class HFGenerator:
    def __init__(self, model_name="google/flan-t5-base", max_input_tokens=1024, max_output_tokens=256):
        print(f"[INFO] Loading generation model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens

    def build_prompt(self, query, retrieved_chunks):
        context_texts = "\n\n".join([f"[Source p.{r['page']}] {r['text']}" for r in retrieved_chunks])
        prompt = (
            "Tu es un assistant expert du Code de la Route français.\n"
            "Réponds à la question suivante en t'appuyant uniquement sur les extraits donnés.\n\n"
            f"Question : {query}\n\n"
            f"Extraits :\n{context_texts}\n\n"
            "Réponse :"
        )
        return prompt

    def generate(self, query, retrieved_chunks):
        prompt = self.build_prompt(query, retrieved_chunks)
        response = self.pipe(
            prompt,
            max_new_tokens=self.max_output_tokens,
            truncation=True,
        )[0]["generated_text"]
        return response


if __name__ == "__main__":
    from src.retrieval.retriever import RAGRetriever

    retriever = RAGRetriever()
    generator = HFGenerator()

    query = "Quelles sont les règles concernant le permis à points ?"
    retrieved = retriever.retrieve(query, top_k=3)
    answer = generator.generate(query, retrieved)

    print("\n=== Réponse ===")
    print(answer)
