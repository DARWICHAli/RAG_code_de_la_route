
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def generate_answer(question, contexts, system_prompt):
    context_text = "\n\n".join([f"[{i+1}] (Page {c['page']}): {c['text']}" for i,c in enumerate(contexts)])
    prompt = system_prompt + "\n\nContext:\n" + context_text + "\n\nUser: " + question
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt, max_tokens=512, temperature=0.0)
    return resp.output_text
