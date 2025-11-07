from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

class HuggingFaceGenerator:
    def __init__(self, model_name="google/flan-t5-base", device=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[INFO] Loaded generation model: {model_name} on {self.device}")

    def _build_prompt(self, question, retrieved, system_prompt=None):
        context = "\n\n".join([f"[Source p.{r['page']}] {r['text']}" for r in retrieved])
        return (
            (system_prompt or "Réponds uniquement à partir du contexte suivant :\n\n")
            + f"{context}\n\nQuestion: {question}\nRéponse:"
        )

    def generate(self, question, retrieved, system_prompt=None,
                 max_new_tokens=256, temperature=0.7, top_p=0.95):
        prompt = self._build_prompt(question, retrieved, system_prompt)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
