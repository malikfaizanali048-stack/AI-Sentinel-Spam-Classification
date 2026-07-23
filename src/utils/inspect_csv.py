import pandas as pd
import sys

print("Script started", flush=True)

try:
    df = pd.read_csv(r"data/raw/spamassassin.csv")
    print("Shape:", df.shape, flush=True)
    print("Columns:", df.columns.tolist(), flush=True)
    print(df.head(3), flush=True)
except Exception as e:
    print("ERROR:", e, flush=True)