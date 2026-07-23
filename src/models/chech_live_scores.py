import joblib
import sys
sys.path.append("src")
from models.embed_text import load_embedding_model, generate_embeddings
from models.score_anomaly import load_autoencoder, compute_anomaly_score
from features.extract_features import extract_all_features
import numpy as np
import pandas as pd

classifier = joblib.load("data/processed/lgbm_model_enhanced.pkl")
embed_model = load_embedding_model()
autoencoder, scaler = load_autoencoder()

test_cases = [
    ("hello", "hello"),
    ("Newsletter", "Check out our latest updates and features this week."),
    ("Your application status", "Your application has been accepted for the internship role."),
    ("Newsletter with links", """Check out our latest updates!
    Read more: https://example.com/blog/post1
    Unsubscribe: https://example.com/unsubscribe?id=12345
    View in browser: https://example.com/view?id=67890
    Follow us: https://twitter.com/example https://linkedin.com/company/example
    """),
]

for subject, body in test_cases:
    features = extract_all_features({"subject": subject, "body": body, "headers": {}, "from": ""})
    emb = generate_embeddings([f"{subject} {body}"], embed_model)[0]
    features["anomaly_score"] = compute_anomaly_score(emb.astype(np.float32), autoencoder, scaler)
    features.update({f"emb_{i}": v for i, v in enumerate(emb)})
    X = pd.DataFrame([features])
    prob = classifier.predict(X)[0]
    print(f"'{subject}' -> {prob:.4f}")

    test_cases_with_headers = [
    {
        "subject": "Newsletter update",
        "body": "Check out our latest features and updates this week. Read more on our blog.",
        "headers": {"Reply-To": "support@notifications.thinkific.com"},
        "from": "noreply@thinkific.com"
    },
]

for case in test_cases_with_headers:
    features = extract_all_features(case)
    emb = generate_embeddings([f"{case['subject']} {case['body']}"], embed_model)[0]
    anomaly = compute_anomaly_score(emb.astype(np.float32), autoencoder, scaler)
    features["anomaly_score"] = anomaly
    features.update({f"emb_{i}": v for i, v in enumerate(emb)})
    X = pd.DataFrame([features])
    prob = classifier.predict(X)[0]
    print(f"'{case['subject']}' -> {prob:.4f} | from_reply_mismatch={features['from_reply_mismatch']} | anomaly_score={anomaly:.4f}")