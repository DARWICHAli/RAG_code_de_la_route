# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from src.retrieval.retriever import RAGRetriever
# from src.generation.generate import RAGGenerator
# from src.safety.filters import is_safe_question, sanitize_response

# app = FastAPI(title="RAG Code de la Route API", version="1.0")

# # Initialisation des composants
# retriever = RAGRetriever(index_path="data/index/faiss.index")
# generator = RAGGenerator(model_name="plguillou/t5-base-fr-sum-cnndm")

# SYSTEM_PROMPT = (
#     "Tu es un assistant français expert du Code de la route. "
#     "Réponds uniquement à partir des passages fournis et cite les pages sources. "
#     "Utilise un ton professionnel et reste toujours en français."
# )

# class QueryReq(BaseModel):
#     question: str

# @app.post("/chat")
# def chat(req: QueryReq):
#     """API endpoint principal : question → réponse RAG."""
#     question = req.question.strip()

#     # Vérification de sécurité
#     if not is_safe_question(question):
#         raise HTTPException(status_code=400, detail="Question non conforme. Reformulez votre demande.")

#     # Récupération des passages pertinents
#     retrieved = retriever.retrieve(question, top_k=5)
#     if not retrieved:
#         raise HTTPException(status_code=404, detail="Aucune information trouvée dans la base.")

#     # Génération de la réponse
#     answer = generator.generate(question, retrieved, SYSTEM_PROMPT)
#     answer = sanitize_response(answer)

#     # Construction de la réponse API
#     return {
#         "question": question,
#         "answer": answer,
#         "sources": [{"page": r["page"], "score": r["score"]} for r in retrieved],
#     }


# @app.get("/")
# def home():
#     return {"message": "Bienvenue sur l'API RAG — Code de la Route 🇫🇷"}



import time
import json
import logging
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from src.retrieval.retriever import RAGRetriever
from src.generation.generate import RAGGenerator
from src.safety.filters import is_safe_question, sanitize_response

# === Configuration du logging ===
logging.basicConfig(
    filename="logs/api_requests.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

app = FastAPI(title="RAG Code de la Route API", version="1.1")

# === Initialisation des composants ===
retriever = RAGRetriever(index_path="data/index/faiss.index")
generator = RAGGenerator(model_name="plguillou/t5-base-fr-sum-cnndm")

SYSTEM_PROMPT = (
    "Tu es un assistant français expert du Code de la route. "
    "Réponds uniquement à partir des passages fournis et cite les pages sources. "
    "Utilise un ton professionnel et reste toujours en français."
)


# === Middleware de journalisation ===
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Journalisation simple
    logging.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={process_time:.2f}s"
    )
    return response


# === Modèle d'entrée ===
class QueryReq(BaseModel):
    question: str


# === Endpoint principal ===
@app.post("/chat")
def chat(req: QueryReq):
    """Endpoint de génération RAG — répond aux questions sur le Code de la route."""
    question = req.question.strip()

    if not is_safe_question(question):
        raise HTTPException(status_code=400, detail="Question non conforme. Reformulez votre demande.")

    start = time.time()
    retrieved = retriever.retrieve(question, top_k=5)
    if not retrieved:
        raise HTTPException(status_code=404, detail="Aucune information trouvée dans la base.")

    answer = generator.generate(question, retrieved, SYSTEM_PROMPT)
    answer = sanitize_response(answer)
    latency = time.time() - start

    avg_score = sum(r["score"] for r in retrieved) / len(retrieved)

    # === Enregistrement détaillé dans les logs ===
    log_entry = {
        "question": question,
        "answer_excerpt": answer[:120],
        "latency_s": round(latency, 2),
        "avg_score": round(avg_score, 3),
        "sources": [r["page"] for r in retrieved],
        "model": generator.model_name,
    }
    logging.info(json.dumps(log_entry, ensure_ascii=False))

    return {
        "question": question,
        "answer": answer,
        "sources": [{"page": r["page"], "score": r["score"]} for r in retrieved],
        "latency_s": round(latency, 2),
    }


@app.get("/")
def home():
    return {
        "message": "Bienvenue sur l'API RAG — Code de la Route 🇫🇷",
        "endpoints": {
            "POST /chat": "Poser une question sur le Code de la route",
        },
    }
