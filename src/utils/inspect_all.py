import pandas as pd

files = {
    "cybercop": "data/raw/cybercop/Phishing_Email.csv",
    "enron": "data/raw/eron/enron_spam_data.csv",   # confirm exact filename
    "nazario": "data/raw/nazario/Nazario_5.csv",
    "ephishing_enron": "data/raw/ephishing/Al-Subaiey_Enron.csv",
    "ephishing_spamassassin": "data/raw/ephishing/Al-Subaiey_SpamAssassin.csv",
}

for name, path in files.items():
    print(f"\n{'='*50}\n{name} ({path})\n{'='*50}")
    try:
        df = pd.read_csv(path, nrows=5)
        print("Columns:", df.columns.tolist())
        print(df.head(2))
    except Exception as e:
        print("ERROR:", e)