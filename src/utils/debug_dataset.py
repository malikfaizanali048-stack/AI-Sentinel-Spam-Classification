import pandas as pd

df = pd.read_csv("data/processed/features_with_embeddings.csv")

print("Shape:", df.shape)
print("\nLabel distribution:\n", df['label'].value_counts())

print("\nNaN counts per column (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nSample spam row (label=1):")
spam_row = df[df['label'] == 1].iloc[0]
print("Subject:", spam_row['subject'])
print("num_urls:", spam_row['num_urls'])
print("urgency_word_count:", spam_row['urgency_word_count'])
print("emb_0:", spam_row['emb_0'])

print("\nSample ham row (label=0):")
ham_row = df[df['label'] == 0].iloc[0]
print("Subject:", ham_row['subject'])
print("num_urls:", ham_row['num_urls'])
print("urgency_word_count:", ham_row['urgency_word_count'])
print("emb_0:", ham_row['emb_0'])

print("\nDo embeddings actually differ between rows?")
print(df[['emb_0','emb_1','emb_2']].describe())

print("\nAre 'from'/'headers' columns present in dataset? (should NOT be)")
print([c for c in df.columns if c in ['from', 'headers']])