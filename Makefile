# ============================================
# 🚦 RAG Code de la Route — Makefile
# ============================================

.PHONY: all ingest index rag serve eval clean

# === Configuration ===
PDF=data/raw/code_de_la_route.pdf
CHUNKS=data/processed/chunks.jsonl
INDEX_DIR=data/index/faiss.index
MODEL_EMBED=sentence-transformers/all-MiniLM-L6-v2
#MODEL_GEN=google/flan-t5-base
MODEL_GEN=plguillou/t5-base-fr-sum-cnndm
QUERY=Que faire en cas d'accident ?

_QUERY := $(if $(QUERY),$(QUERY),$(DEFAULT_QUERY))

# === Default target ===
all: ingest index rag

# ============================================
# 🧩 Data ingestion (PDF → chunks)
# ============================================
ingest:
	@echo "[INFO] 📘 Ingesting PDF → chunks..."
	python -m src.data.ingest_pdf \
		--pdf $(PDF) \
		--out_dir data/processed \
		--chunk_size 1000 \
		--overlap 200 \
		--plan_pages 3 6

# ============================================
# 🧠 Indexing (chunks → FAISS index)
# ============================================
index:
	@echo "[INFO] 🔍 Building FAISS index..."
# 	python -m src.data.indexing \
# 		--chunks $(CHUNKS) \
# 		--out_dir $(INDEX_DIR) \
# 		--model_name $(MODEL_EMBED)
	python -m src.data.indexing \
		--chunks $(CHUNKS) \
		--model $(MODEL_EMBED) \
		--index_path $(INDEX_DIR)




# ============================================
# 🤖 RAG generation (retrieval + generation)
# ============================================
rag:
	@echo "[STEP] Génération RAG (retrieval + génération Hugging Face)..."
# 	python -m src.generation.generate \
# 		--query "$(_QUERY)" \
# 		--retriever_model $(MODEL_EMBED) \
# 		--generator_model $(MODEL_GEN) \
# 		--top_k 5
	python -m src.generation.generate \
		--QUERY "$(_QUERY)" \
		--model $(MODEL_GEN) \
		--index_path $(INDEX_DIR) \
		--top_k 5


# ============================================
# 🌐 API serving
# ============================================
serve:
	@echo "[INFO] 🚀 Starting FastAPI server..."
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

# ============================================
# 🧪 Evaluation
# ============================================
eval:
	@echo "[INFO] 📊 Running evaluation pipeline..."
# 	python -m src.eval.evaluate \
# 		--config experiments/exp_rag_default.yaml
	python src/eval/evaluate.py \
		--model_name google/flan-t5-base 
		--index_path $(INDEX) 
		--holdout_path $(HOLDOUT)

# ============================================
# 🧹 Clean artifacts
# ============================================
clean:
	@echo "[INFO] 🧼 Cleaning generated artifacts..."
	rm -rf data/processed/* data/index/* mlruns/

# ============================================
# 🧹 Clean logs
# ============================================

clean-logs:
	@echo "[INFO] 🧼 Cleaning log files..."
	rm -f logs/api_requests.log
