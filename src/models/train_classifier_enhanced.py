"""
Trains an ENHANCED model on (baseline train_final.csv + 70% of E-PhishLLM).
Saves to a SEPARATE file -- does NOT overwrite your baseline lgbm_model.pkl.

Place this file at: src/models/train_classifier_enhanced.py
Run from project root, AFTER process_ephishllm_split.py:
    python -u -m src.models.train_classifier_enhanced
"""

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import joblib


def load_and_combine(
    baseline_path="data/processed/train_final.csv",
    ephishllm_train_path="data/processed/ephishllm_train_final.csv",
):
    baseline_df = pd.read_csv(baseline_path, low_memory=False)
    eph_df = pd.read_csv(ephishllm_train_path, low_memory=False)

    print("Baseline train shape:", baseline_df.shape)
    print("E-PhishLLM 70% slice shape:", eph_df.shape)

    combined = pd.concat([baseline_df, eph_df], ignore_index=True)
    combined = combined.dropna(subset=["body"])
    print("Combined enhanced train shape:", combined.shape)
    print("Label distribution:", dict(combined["label"].value_counts()))
    return combined


def prepare_data(df):
    drop_cols = ["subject", "body", "label"]
    X = df.drop(columns=drop_cols)
    y = df["label"]
    return X, y


def train_enhanced_model():
    df = load_and_combine()
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Same hyperparameters as your baseline model, for a fair comparison
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "verbose": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)],
    )

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\n--- Enhanced Model Internal Validation (20% held from combined train) ---")
    print(classification_report(y_test, y_pred, target_names=["Not Spam", "Spam"]))
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))
    print("PR-AUC:", average_precision_score(y_test, y_pred_prob))

    joblib.dump(model, "data/processed/lgbm_model_enhanced.pkl")
    print("\nEnhanced model saved to: data/processed/lgbm_model_enhanced.pkl")
    print("Baseline model at data/processed/lgbm_model.pkl was NOT touched.")

    return model


if __name__ == "__main__":
    train_enhanced_model()