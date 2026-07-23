import pandas as pd
import numpy as np
import sys
sys.path.append("src")
from features.extract_features import extract_all_features
from models.embed_text import load_embedding_model, generate_embeddings
from models.score_anomaly import load_autoencoder, compute_anomaly_score

def process_file(input_path, output_path, embed_model, autoencoder, scaler):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['body']).reset_index(drop=True)
    print(f"Processing {input_path}: {df.shape}")

    # Features
    feature_rows = df.apply(lambda row: extract_all_features({
        "subject": row.get("subject", ""),
        "body": row.get("body", ""),
        "headers": {},
        "from": ""
    }), axis=1)
    feature_df = pd.DataFrame(list(feature_rows))

    # Embeddings
    texts = (df['subject'].fillna('') + " " + df['body'].fillna('')).tolist()
    embeddings = generate_embeddings(texts, embed_model)
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])

    # Anomaly scores
    print("Computing anomaly scores...")
    scores = [compute_anomaly_score(embeddings[i].astype(np.float32), autoencoder, scaler) for i in range(len(embeddings))]

    final_df = pd.concat([df[['subject', 'body', 'label']], feature_df, emb_df], axis=1)
    final_df['anomaly_score'] = scores

    final_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} {final_df.shape}\n")

if __name__ == "__main__":
    embed_model = load_embedding_model()
    autoencoder, scaler = load_autoencoder()

    process_file("data/processed/train_raw.csv", "data/processed/train_final.csv", embed_model, autoencoder, scaler)
    process_file("data/processed/test_nazario_raw.csv", "data/processed/test_nazario_final.csv", embed_model, autoencoder, scaler)
    process_file("data/processed/test_ephishllm_raw.csv", "data/processed/test_ephishllm_final.csv", embed_model, autoencoder, scaler)