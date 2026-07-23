import pandas as pd
import numpy as np
from src.models.score_anomaly import load_autoencoder, compute_anomaly_score

def add_anomaly_scores():
    df = pd.read_csv("data/processed/features_with_embeddings.csv")
    df = df.dropna(subset=['body']).reset_index(drop=True)

    model, scaler = load_autoencoder()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    print("Computing anomaly scores for all rows...")
    scores = []
    for i, row in df.iterrows():
        emb = row[emb_cols].values.astype(np.float32)
        score = compute_anomaly_score(emb, model, scaler)
        scores.append(score)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(df)} done")

    df['anomaly_score'] = scores
    df.to_csv("data/processed/final_dataset.csv", index=False)
    print("Saved final dataset:", df.shape)

if __name__ == "__main__":
    add_anomaly_scores()