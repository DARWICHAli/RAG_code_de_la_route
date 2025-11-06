# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.generation.generate import run_rag_pipeline
from src.safety.filters import is_safe_question

app = FastAPI()

SYSTEM_PROMPT = (
    "Tu es un assistant expert du Code de la route français. "
    "Utilise uniquement les passages fournis et cite la page source."
)

class QueryReq(BaseModel):
    question: str

@app.post("/ask")
def ask(req: QueryReq):
    question = req.question.strip()
    
    # 1️⃣ Safety check avant pipeline
    if not is_safe_question(question):
        raise HTTPException(
            status_code=400,
            detail="Question hors-sujet détectée ! Je ne peux répondre qu'au Code de la Route."
        )

    # 2️⃣ Run full RAG pipeline
    answer = run_rag_pipeline(
        query=question,
        retriever_model="sentence-transformers/all-MiniLM-L6-v2",  # configurable
        generator_model="google/flan-t5-base",                     # configurable
        top_k=5
    )

    return {"answer": answer}


@app.get("/health")
def health():
    return {"status": "ok",
            "retriever_model": "sentence-transformers/all-MiniLM-L6-v2",
            "generator_model": "google/flan-t5-base"}