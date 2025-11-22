# 🇫🇷 RAG — Code de la Route (Données officielles Légifrance)
Ce projet implémente un **système RAG (Retrieval-Augmented Generation)** spécialisé dans le **Code de la route français**, basé sur des données **officielles LÉGIFRANCE** récupérées automatiquement via leur API.

---

## 🚀 Fonctionnalités principales

- 🔎 **Recherche intelligente** dans le Code de la route

- 🤖 **Génération de réponses** en français via un modèle HF (ex : plguillou/t5-base-fr-sum-cnndm)

- 📚 **Contextualisation automatique** grâce à des chunks optimisés

- ⚖️ **Précision juridique** et citations de pages / articles

- 🛡️ **Filtrage de sécurité** pour garantir des réponses fiables

- 🌐 **API FastAPI** pour accéder au chatbot

- 🚀 **Pipeline reproductible** (Makefile, configs, ML-friendly)


---

## 📁 Structure du projet

```
.
├── configs/
├── data/
│   ├── raw/                # PDF original
│   ├── processed/          # Chunks JSONL après ingestion
│   └── index/              # Index FAISS
├── src/
│   ├── api/                # FastAPI
│   ├── data/               # Ingestion / Indexation
│   ├── retrieval/          # Retriever FAISS
│   ├── generation/         # Génération HuggingFace
│   └── safety/             # Filtres de sécurité
├── experiments/
├── Makefile
└── README.md
```

---

## 🔧 Installation

```bash
git clone https://github.com/DARWICHAli/RAG_code_de_la_route
cd RAG_code_de_la_route
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏗️ Pipeline complet

### 1️⃣ Ingestion du PDF  —on a plus besoin!
```bash
make ingest
```

### 2️⃣ Indexation FAISS
```bash
make index
```

### 3️⃣ Lancer l'API RAG
```bash
make serve
```

API disponible via :

```
POST /chat
{
  "question": "Quel est le délai pour récupérer des points ?"
}
```

---

## 🤖 Modèles utilisés

| Composant | Modèle | Description |
|----------|--------|-------------|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | compact + performant |
| **Génération** | `google/flan-t5-base` | excellent sur textes français |
| **Index** | FAISS L2 | rapide et scalable | 

---

## 🧪 Tests (pipeline complet)

```bash
python src/generation/generate.py   --question "Qu'est‑ce que le permis probatoire ?"   --top_k 3
```
ou
```bash
make rag
```


---

## 📦 Déploiement

Serveur Uvicorn :
```bash
make serve
```

Déploiement Docker (optionnel) :
```bash
docker build -t rag-code-route .
docker run -p 8000:8000 rag-code-route
```

---

## 🔮 Améliorations futures
- Passage à un modèle FR plus puissant (LLaMA 3 FR, Mistral FR).
- Amélioration du chunking contextuel (titres, articles, chapitres).
- Détection des articles juridiques pour structurer les sources.
- Historique de conversation dans l’API.
- Fine‑tuning supervisé (LoRA) sur Q/A Code de la route.

---

## 👤 Auteur
**Ali Darwich**  
Data Scientist — France 🇫🇷  
Projet full open-source.

---

## ⭐ Contribute
Les PR et issues sont les bienvenues !

---
