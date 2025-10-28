.PHONY: all ingest index serve eval

PDF=data/raw/code_de_la_route.pdf

all: ingest index

ingest:
	python src/data/ingest_pdf.py --pdf $(PDF) --out_dir data/processed --chunk_size 1000 --overlap 200

index:
	python src/data/indexing.py --chunks data/processed/chunks.jsonl --model all-MiniLM-L6-v2 --index_path data/index/faiss.index

serve:
	uvicorn src.api.app:app --host 0.0.0.0 --port 8000

eval:
	python src/eval/evaluate.py --config experiments/exp_rag_default.yaml
