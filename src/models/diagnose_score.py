import pandas as pd
import joblib

model = joblib.load("data/processed/lgbm_model.pkl")

def check_score_distribution(name, path):
    df = pd.read_csv(path, low_memory=False)
    df = df.dropna(subset=['body'])
    drop_cols = ['subject', 'body', 'label']
    X = df.drop(columns=drop_cols)
    probs = model.predict(X)

    print(f"\n{name} score distribution:")
    print(pd.Series(probs).describe())
    print("Below 0.3:", (probs < 0.3).mean())
    print("Below 0.5:", (probs < 0.5).mean())
    print("Below 0.7:", (probs < 0.7).mean())

check_score_distribution("Nazario", "data/processed/test_nazario_final.csv")
check_score_distribution("E-PhishLLM", "data/processed/test_ephishllm_final.csv")