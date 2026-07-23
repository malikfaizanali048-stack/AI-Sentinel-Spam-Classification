"""
Runs ephishllm_train_raw.csv and ephishllm_holdout_raw.csv through the exact
same feature extraction / embedding / anomaly-score pipeline used for your
baseline data, by reusing process_file() from utils.process_dataset.

This guarantees the enhanced-model features are computed identically to the
baseline model's features -- same embed model, same fitted autoencoder/scaler.

Place this file at: src/utils/process_ephishllm_split.py
Run from project root, AFTER split_ephishllm.py:
    python -u -m src.utils.process_ephishllm_split
"""

import sys
sys.path.append("src")

from models.embed_text import load_embedding_model
from models.score_anomaly import load_autoencoder
from utils.process_dataset import process_file


def main():
    embed_model = load_embedding_model()
    autoencoder, scaler = load_autoencoder()

    process_file(
        "data/processed/ephishllm_train_raw.csv",
        "data/processed/ephishllm_train_final.csv",
        embed_model, autoencoder, scaler,
    )
    process_file(
        "data/processed/ephishllm_holdout_raw.csv",
        "data/processed/ephishllm_holdout_final.csv",
        embed_model, autoencoder, scaler,
    )


if __name__ == "__main__":
    main()