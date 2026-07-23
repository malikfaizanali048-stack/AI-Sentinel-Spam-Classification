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
    print(f"EXTERNAL EVALUATION: {name}")
    print(f"{'='*60}")
    print(f"Samples: {len(y)}  |  Label distribution: {dict(y.value_counts())}")

    try:
        print(classification_report(y, y_pred, target_names=["Not Spam/Legit", "Spam/Phishing"]))
        print("ROC-AUC:", roc_auc_score(y, y_pred_prob))
        print("PR-AUC:", average_precision_score(y, y_pred_prob))
    except ValueError as e:
        # Handles Nazario's case: only one class present, so ROC-AUC/classification report differ
        print("Note: dataset contains only one class, reporting recall/detection rate instead")
        detection_rate = (y_pred == y).mean()
        print(f"Detection rate (recall on phishing-only set): {detection_rate:.4f}")

if __name__ == "__main__":
    model = joblib.load("data/processed/lgbm_model.pkl")

    evaluate_on_dataset("Nazario (classic phishing)", "data/processed/test_nazario_final.csv", model)
    evaluate_on_dataset("E-PhishLLM (modern LLM-generated phishing)", "data/processed/test_ephishllm_final.csv", model)