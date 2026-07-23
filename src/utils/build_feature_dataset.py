import pandas as pd
import sys
sys.path.append("src")

from utils.load_spamassassin import load_spamassassin
from features.extract_features import extract_all_features

def build_dataset():
    df = load_spamassassin()
    print("Loaded:", df.shape)

    feature_rows = df.apply(extract_all_features, axis=1)
    feature_df = pd.DataFrame(list(feature_rows))

    final_df = pd.concat([df[['subject', 'body', 'label']], feature_df], axis=1)
    final_df.to_csv("data/processed/features_dataset.csv", index=False)
    print("Saved feature dataset:", final_df.shape)
    print(final_df.head())

if __name__ == "__main__":
    build_dataset()