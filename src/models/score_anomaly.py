import torch
import joblib
import numpy as np
from src.models.train_autoencoder import Autoencoder

def load_autoencoder(input_dim=384):
    model = Autoencoder(input_dim=input_dim)
    model.load_state_dict(torch.load("data/processed/autoencoder.pt"))
    model.eval()

    scaler = joblib.load("data/processed/autoencoder_scaler.pkl")
    return model, scaler

def compute_anomaly_score(embedding, model, scaler):
    """embedding: 1D numpy array of shape (384,)"""
    embedding_scaled = scaler.transform(embedding.reshape(1, -1))
    embedding_t = torch.tensor(embedding_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = model(embedding_t)
        error = torch.mean((reconstructed - embedding_t) ** 2).item()

    return error

if __name__ == "__main__":
    model, scaler = load_autoencoder()

    # Quick test using a real embedding from your dataset
    import pandas as pd
    df = pd.read_csv("data/processed/features_with_embeddings.csv")
    df = df.dropna(subset=['body'])
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    ham_sample = df[df['label'] == 0].iloc[0][emb_cols].values.astype(np.float32)
    spam_sample = df[df['label'] == 1].iloc[0][emb_cols].values.astype(np.float32)

    ham_score = compute_anomaly_score(ham_sample, model, scaler)
    spam_score = compute_anomaly_score(spam_sample, model, scaler)

    print("Ham anomaly score:", ham_score)
    print("Spam anomaly score:", spam_score)