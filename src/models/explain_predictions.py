import pandas as pd
import joblib
import shap
import numpy as np

def load_model_and_data():
    model = joblib.load("data/processed/lgbm_model_enhanced.pkl")
    df = pd.read_csv("data/processed/final_dataset.csv")
    df = df.dropna(subset=['body']).reset_index(drop=True)
    return model, df

def get_readable_feature_names():
    """Map raw feature names to human-readable labels for SHAP output."""
    return {
        "num_urls": "Number of links in email",
        "num_ip_urls": "Links using raw IP address",
        "suspicious_tld_count": "Suspicious domain extensions",
        "avg_url_length": "Average link length",
        "spf_pass": "SPF authentication passed",
        "dkim_pass": "DKIM authentication passed",
        "dmarc_pass": "DMARC authentication passed",
        "from_reply_mismatch": "Sender/Reply-To mismatch",
        "urgency_word_count": "Urgency language detected",
        "exclamation_count": "Excessive punctuation",
        "subject_caps_ratio": "ALL CAPS usage in subject",
        "anomaly_score": "Unusual pattern vs normal mail"
    }

def explain_single_prediction(row_index, model, df):
    drop_cols = ['subject', 'body', 'label']
    X = df.drop(columns=drop_cols)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # For a single row, get feature contributions
    row_shap = shap_values[row_index]
    feature_names = X.columns.tolist()

    readable_names = get_readable_feature_names()

    # Only show non-embedding features in the explanation (embeddings aren't human-readable)
    contributions = []
    for name, value in zip(feature_names, row_shap):
        if name.startswith("emb_"):
            continue  # skip raw embedding dims, not interpretable
        contributions.append((readable_names.get(name, name), value))

    # Sort by absolute impact, show top reasons
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n--- Explanation for row {row_index} ---")
    print(f"Actual label: {'Spam' if df.iloc[row_index]['label'] == 1 else 'Not Spam'}")
    print(f"Model prediction: {model.predict(X.iloc[[row_index]])[0]:.4f}")
    print("\nTop reasons:")
    for name, value in contributions[:5]:
        direction = "increases spam likelihood" if value > 0 else "decreases spam likelihood"
        print(f"  {name}: {direction} (impact: {value:.4f})")

if __name__ == "__main__":
    model, df = load_model_and_data()

    spam_idx = df[df['label'] == 1].index[0]
    ham_idx = df[df['label'] == 0].index[0]

    explain_single_prediction(spam_idx, model, df)
    explain_single_prediction(ham_idx, model, df)