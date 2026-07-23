import time
print("STEP 1: starting imports", flush=True)
import numpy as np
import pandas as pd
import joblib
import torch
print("STEP 2: base imports done", flush=True)
from src.ingestion.fetch import fetch_new_emails_since
print("STEP 3: fetch imported", flush=True)
from src.features.extract_features import extract_all_features
print("STEP 4: features imported", flush=True)
from src.models.embed_text import load_embedding_model, generate_embeddings
print("STEP 5: embed_text imported", flush=True)
from src.models.score_anomaly import load_autoencoder, compute_anomaly_score
print("STEP 6: score_anomaly imported", flush=True)

POLL_INTERVAL_SECONDS = 45

def load_all_models():
    print("Loading models...")
    classifier = joblib.load("data/processed/lgbm_model_enhanced.pkl")
    embed_model = load_embedding_model()
    autoencoder, scaler = load_autoencoder()
    print("All models loaded.\n")
    return classifier, embed_model, autoencoder, scaler

def process_email(email_data, classifier, embed_model, autoencoder, scaler):
    subject = email_data.get("subject") or ""
    body = email_data.get("body") or ""
    headers = email_data.get("headers") or {}
    sender = email_data.get("from") or ""

    row = {"subject": subject, "body": body, "headers": headers, "from": sender}
    features = extract_all_features(row)

    text = f"{subject} {body}"
    embedding = generate_embeddings([text], embed_model)[0]

    # Embeddings added first (emb_0 to emb_383)
    emb_dict = {f"emb_{i}": v for i, v in enumerate(embedding)}
    features.update(emb_dict)

    # Anomaly score added last to match training feature order
    anomaly_score = compute_anomaly_score(embedding.astype(np.float32), autoencoder, scaler)
    features["anomaly_score"] = anomaly_score

    X = pd.DataFrame([features])

    # Reorder columns explicitly using the model's exact expected order
    if hasattr(classifier, "feature_name"):
        expected_cols = classifier.feature_name()
        X = X.reindex(columns=expected_cols, fill_value=0.0)

    # Raw LightGBM Booster predict outputs the probability float directly
    prediction_prob = float(classifier.predict(X)[0])

    if prediction_prob > 0.75:
        label = "SPAM"
    elif prediction_prob > 0.5:
        label = "SUSPICIOUS"
    else:
         label = "NOT SPAM"

    return label, prediction_prob

def run_loop():
    classifier, embed_model, autoencoder, scaler = load_all_models()
    last_uid = None

    while True:
        try:
            emails = fetch_new_emails_since(last_uid)
            for email_data in emails:
                label, prob = process_email(email_data, classifier, embed_model, autoencoder, scaler)
                print(f"[{label}] ({prob:.3f}) Subject: {email_data.get('subject')}")
                last_uid = email_data["uid"]
        except Exception as e:
            print("Error during polling cycle:", e)

        time.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_loop()