# src/generation/generate_hf.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class HuggingFaceGenerator:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.3", device_map="auto"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
        )

    def generate(self, question, contexts, system_prompt):
        context_text = "\n\n".join([f"[{i+1}] (Page {c['page']}): {c['text']}" for i,c in enumerate(contexts)])
        prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context_text}\n\n"
            f"User: {question}\nAssistant:"
        )
        output = self.pipe(prompt, num_return_sequences=1)
        return output[0]["generated_text"].split("Assistant:")[-1].strip()
