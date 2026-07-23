import sys
sys.path.append("src")
import joblib
import shap
import pandas as pd
import numpy as np
from models.embed_text import load_embedding_model, generate_embeddings
from models.score_anomaly import load_autoencoder, compute_anomaly_score
from features.extract_features import extract_all_features
from ingestion.fetch import fetch_recent_emails

classifier = joblib.load("data/processed/lgbm_model_enhanced.pkl")
embed_model = load_embedding_model()
autoencoder, scaler = load_autoencoder()
explainer = shap.TreeExplainer(classifier)

readable_names = {
    "num_urls": "Number of links", "num_ip_urls": "IP-based links",
    "suspicious_tld_count": "Suspicious domains", "avg_url_length": "Avg link length",
    "spf_pass": "SPF pass", "dkim_pass": "DKIM pass", "dmarc_pass": "DMARC pass",
    "from_reply_mismatch": "Sender/Reply-To mismatch", "urgency_word_count": "Urgency language",
    "exclamation_count": "Excessive punctuation", "subject_caps_ratio": "ALL CAPS subject",
    "anomaly_score": "Unusual pattern vs normal mail"
}

emails = fetch_recent_emails(limit=10)

print("\n--- RAW FETCHED EMAILS ---")
for i, e in enumerate(emails):
    print(f"{i}: {e.get('subject')}")
print("--- END RAW LIST ---\n")

for e in emails:
    subject = e.get("subject", "")
    body = e.get("body", "")
    features = extract_all_features(e)
    emb = generate_embeddings([f"{subject} {body}"], embed_model)[0]
    features["anomaly_score"] = compute_anomaly_score(emb.astype(np.float32), autoencoder, scaler)
    features.update({f"emb_{i}": v for i, v in enumerate(emb)})
    X = pd.DataFrame([features])
    prob = classifier.predict(X)[0]

    print(f"Subject: {subject[:60]:<60} Score: {prob:.4f}")

    if prob > 0.3:
        shap_vals = explainer.shap_values(X)
        row_shap = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
        contributions = [(readable_names.get(n, n), v) for n, v in zip(X.columns, row_shap) if not n.startswith("emb_")]
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        print("  Top reasons:")
        for name, val in contributions[:6]:
            print(f"    {name}: {val:+.4f}")
        print()