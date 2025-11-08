# Step 1: Install required packages
# pip install sentence-transformers scikit-learn numpy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Load a sentence transformer model ---
model = SentenceTransformer('all-MiniLM-L6-v2')  # small and fast

# --- Example documents and queries ---
documents = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a popular programming language for data science.",
    "OpenAI develops powerful language models like GPT."
]

queries = [
    "Where is the Eiffel Tower?",
    "What programming language is used for data science?",
    "Who makes GPT models?"
]

# --- Generate embeddings ---
doc_embeddings = model.encode(documents)
query_embeddings = model.encode(queries)

# --- Compute cosine similarity and retrieve top matches ---
for i, q_emb in enumerate(query_embeddings):
    sims = cosine_similarity([q_emb], doc_embeddings)[0]
    top_indices = sims.argsort()[::-1]  # descending order
    print(f"\nQuery: {queries[i]}")
    for rank, idx in enumerate(top_indices[:3], start=1):
        print(f"Top {rank}: Doc {idx} (score: {sims[idx]:.3f})")
        print(f"Text: {documents[idx]}")
