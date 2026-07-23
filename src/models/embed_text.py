import os
import numpy as np
import pandas as pd

# Set global environment flags to prevent Hugging Face network requests
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_embedding_model():
    print("Loading MiniLM model...")
    try:
        # Pass local_files_only directly as a primary parameter
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception as e:
        print(f"Offline load failed ({e}). Attempting normal download...")
        # Fallback if the cache isn't found locally yet
        model = SentenceTransformer(MODEL_NAME)
    return model

def generate_embeddings(texts, model):
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings

if __name__ == "__main__":
    model = load_embedding_model()
    sample_texts = [
        "Urgent: verify your account now or it will be suspended",
        "Hey, are we still meeting for lunch tomorrow?"
    ]
    embeddings = generate_embeddings(sample_texts, model)
    print("Embedding shape:", embeddings.shape)
    print("First 5 values of embedding 1:", embeddings[0][:5])