import pandas as pd

raw = pd.read_csv("data/raw/nazario/Nazario_5.csv")
final = pd.read_csv("data/processed/test_nazario_final.csv", low_memory=False)

# Both should have same row order/count since we never filtered/shuffled Nazario
final = final.drop(columns=['label'])
final['label'] = raw['label'].values[:len(final)]  # align lengths just in case of dropped NaNs

final.to_csv("data/processed/test_nazario_final.csv", index=False)
print("Fixed label distribution:", final['label'].value_counts())