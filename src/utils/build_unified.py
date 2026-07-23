import pandas as pd
import json
import sys
sys.path.append("src")
from utils.load_spamassassin import load_spamassassin

def load_cybercop(path="data/raw/cybercop/Phishing_Email.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"Email Text": "body", "Email Type": "label_raw"})
    df["subject"] = ""
    df["label"] = df["label_raw"].apply(lambda x: 1 if "phish" in str(x).lower() else 0)
    return df[["subject", "body", "label"]]

def load_enron(path="data/raw/eron/enron_spam_data.csv"):
    df = pd.read_csv(path)
    df = df.rename(columns={"Subject": "subject", "Message": "body", "Spam/Ham": "label_raw"})
    df["label"] = df["label_raw"].apply(lambda x: 1 if str(x).lower() == "spam" else 0)
    return df[["subject", "body", "label"]]

def load_nazario(path="data/raw/nazario/Nazario_5.csv"):
    df = pd.read_csv(path)
    # Use the dataset's OWN label column — do NOT overwrite it
    return df[["subject", "body", "label"]]
def load_ephishllm(path="data/raw/ephishllm/ephishLLM.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df[df["Language"] == "en"]  # English subset only, per paper's protocol
    df = df.rename(columns={"Subject": "subject", "Body": "body", "type": "label"})
    return df[["subject", "body", "label"]]

def build_train_set():
    print("Loading SpamAssassin...")
    sa = load_spamassassin()[["subject", "body", "label"]]
    print("Loading CyberCop...")
    cc = load_cybercop()
    print("Loading Enron...")
    en = load_enron()

    if len(en) > 15000:
        en = en.sample(15000, random_state=42)

    train_df = pd.concat([sa, cc, en], ignore_index=True)
    train_df = train_df.dropna(subset=["body"])
    train_df.to_csv("data/processed/train_raw.csv", index=False)
    print("Train set saved:", train_df.shape)
    print(train_df['label'].value_counts())

def build_test_sets():
    print("Loading Nazario...")
    naz = load_nazario()
    naz.to_csv("data/processed/test_nazario_raw.csv", index=False)
    print("Nazario test saved:", naz.shape)
    print(naz['label'].value_counts())

    print("Loading E-PhishLLM (English)...")
    eph = load_ephishllm()
    eph = eph.dropna(subset=["body"])
    eph.to_csv("data/processed/test_ephishllm_raw.csv", index=False)
    print("E-PhishLLM test saved:", eph.shape)
    print(eph['label'].value_counts())

if __name__ == "__main__":
    build_train_set()
    build_test_sets()