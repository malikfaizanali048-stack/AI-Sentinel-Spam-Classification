import pandas as pd
import joblib
from sklearn.metrics import classification_report

model = joblib.load("data/processed/lgbm_model.pkl")
df = pd.read_csv("data/processed/test_ephishllm_final.csv", low_memory=False)
df = df.dropna(subset=['body'])
X = df.drop(columns=['subject', 'body', 'label'])
y = df['label']
probs = model.predict(X)

for threshold in [0.5, 0.4, 0.3, 0.2, 0.15]:
    preds = (probs > threshold).astype(int)
    print(f"\n--- Threshold: {threshold} ---")
    print(classification_report(y, preds, target_names=["Legit", "Phishing"]))