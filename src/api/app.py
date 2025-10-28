# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retrieval.retriever import Retriever
from src.generation.generate_hf import HuggingFaceGenerator
from src.safety.filters import check_input, should_decline

app = FastAPI()
retriever = Retriever(index_path="data/index/faiss.index")
generator = HuggingFaceGenerator(model_name="mistralai/Mistral-7B-Instruct-v0.3")

SYSTEM_PROMPT = (
    "Tu es un assistant expert du Code de la route français. "
    "Utilise uniquement les passages fournis et cite la page source."
)

class QueryReq(BaseModel):
    question: str

@app.post("/chat")
def chat(req: QueryReq):
    ok, reason = check_input(req.question)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    retrieved = retriever.retrieve(req.question, top_k=5)
    if should_decline(req.question, retrieved):
        return {"answer": "Je ne peux pas répondre avec certitude. Consultez la source officielle.", "sources": []}

    ans = generator.generate(req.question, retrieved, SYSTEM_PROMPT)
    return {
        "answer": ans,
        "sources": [{"page": r["page"], "score": r["score"]} for r in retrieved],
    }
