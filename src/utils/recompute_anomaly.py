import pandas as pd
import numpy as np
import sys
sys.path.append("src")
from models.score_anomaly import load_autoencoder, compute_anomaly_score

def recompute(path):
    df = pd.read_csv(path)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    autoencoder, scaler = load_autoencoder()

    scores = []
    for i, row in df.iterrows():
        emb = row[emb_cols].values.astype(np.float32)
        scores.append(compute_anomaly_score(emb, autoencoder, scaler))

    df['anomaly_score'] = scores
    df.to_csv(path, index=False)
    print(f"Updated anomaly scores in {path}: {df.shape}")

if __name__ == "__main__":
    recompute("data/processed/train_final.csv")
    recompute("data/processed/test_nazario_final.csv")
    recompute("data/processed/test_ephishllm_final.csv")