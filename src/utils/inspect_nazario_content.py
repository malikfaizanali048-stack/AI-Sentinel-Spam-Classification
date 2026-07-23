import pandas as pd

df = pd.read_csv("data/raw/nazario/Nazario_5.csv")
print("Original label column values:", df['label'].unique())
print("\nSample rows (subject + body preview):\n")

for i in range(5):
    print(f"--- Row {i} ---")
    print("Subject:", df.iloc[i]['subject'])
    print("Body preview:", str(df.iloc[i]['body'])[:300])
    print()