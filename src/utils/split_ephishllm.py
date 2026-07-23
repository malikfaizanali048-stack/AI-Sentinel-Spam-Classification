"""
Splits data/processed/test_ephishllm_raw.csv into:
  - 70% -> ephishllm_train_raw.csv   (will be added to training pool)
  - 30% -> ephishllm_holdout_raw.csv (never used in training; true unseen test set)

Place this file at: src/utils/split_ephishllm.py
Run from project root: python -u -m src.utils.split_ephishllm
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_ephishllm(
    input_path="data/processed/test_ephishllm_raw.csv",
    train_out="data/processed/ephishllm_train_raw.csv",
    holdout_out="data/processed/ephishllm_holdout_raw.csv",
    train_frac=0.70,
    random_state=42,
):
    df = pd.read_csv(input_path, low_memory=False)
    df = df.dropna(subset=["body"]).reset_index(drop=True)

    train_df, holdout_df = train_test_split(
        df,
        train_size=train_frac,
        random_state=random_state,
        stratify=df["label"],
    )

    train_df.to_csv(train_out, index=False)
    holdout_df.to_csv(holdout_out, index=False)

    print(f"E-PhishLLM split (seed={random_state}):")
    print(f"  Train slice ({train_frac*100:.0f}%): {train_df.shape} -> {train_out}")
    print(f"    Label distribution: {dict(train_df['label'].value_counts())}")
    print(f"  Holdout slice ({(1-train_frac)*100:.0f}%): {holdout_df.shape} -> {holdout_out}")
    print(f"    Label distribution: {dict(holdout_df['label'].value_counts())}")
    print("\nIMPORTANT: holdout_out is your new unseen test set.")
    print("Never train on it, never tune thresholds on it, never look at it")
    print("until final evaluation of the enhanced model.")


if __name__ == "__main__":
    split_ephishllm()