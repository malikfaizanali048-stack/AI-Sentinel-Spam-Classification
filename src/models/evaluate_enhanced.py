"""
Evaluates BASELINE and ENHANCED models side-by-side:
  - Nazario (full, unchanged) -- both models
  - E-PhishLLM 30% holdout (never seen by either model's training,
    since baseline never saw any E-PhishLLM and enhanced only saw the other 70%)

Your baseline numbers (31% / 68% recall) remain the primary, headline finding.
This script's enhanced results are the secondary mitigation experiment.

Place this file at: src/models/evaluate_enhanced.py
Run from project root, AFTER train_classifier_enhanced.py:
    python -u -m src.models.evaluate_enhanced
"""

import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


def evaluate_on_dataset(name, path, model):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=['body'])

    drop_cols = ['subject', 'body', 'label']
    X = df.drop(columns=drop_cols)
    y = df['label']

    y_pred_prob = model.predict(X)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Samples: {len(y)}  |  Label distribution: {dict(y.value_counts())}")

    try:
        print(classification_report(y, y_pred, target_names=["Not Spam/Legit", "Spam/Phishing"]))
        print("ROC-AUC:", roc_auc_score(y, y_pred_prob))
        print("PR-AUC:", average_precision_score(y, y_pred_prob))
    except ValueError:
        print("Note: dataset contains only one class, reporting detection rate instead")
        detection_rate = (y_pred == y).mean()
        print(f"Detection rate: {detection_rate:.4f}")


if __name__ == "__main__":
    baseline_model = joblib.load("data/processed/lgbm_model.pkl")
    enhanced_model = joblib.load("data/processed/lgbm_model_enhanced.pkl")

    print("\n" + "#" * 60)
    print("# BASELINE MODEL (zero E-PhishLLM exposure) -- PRIMARY FINDING")
    print("#" * 60)
    evaluate_on_dataset("Nazario (baseline)", "data/processed/test_nazario_final.csv", baseline_model)
    evaluate_on_dataset(
        "E-PhishLLM 30% HOLDOUT (baseline, apples-to-apples vs enhanced)",
        "data/processed/ephishllm_holdout_final.csv", baseline_model,
    )
    evaluate_on_dataset(
        "E-PhishLLM FULL SET (baseline, for reference -- includes the 70% enhanced later trains on)",
        "data/processed/test_ephishllm_final.csv", baseline_model,
    )

    print("\n" + "#" * 60)
    print("# ENHANCED MODEL (70% E-PhishLLM exposure) -- SECONDARY / MITIGATION EXPERIMENT")
    print("#" * 60)
    evaluate_on_dataset("Nazario (enhanced)", "data/processed/test_nazario_final.csv", enhanced_model)
    evaluate_on_dataset(
        "E-PhishLLM 30% HOLDOUT (enhanced, true unseen test)",
        "data/processed/ephishllm_holdout_final.csv", enhanced_model,
    )