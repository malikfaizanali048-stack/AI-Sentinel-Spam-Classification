import pandas as pd
import sys
sys.path.append("src")
from models.embed_text import load_embedding_model, generate_embeddings

def add_embeddings_to_dataset():
    df = pd.read_csv("data/processed/features_dataset.csv")
    print("Loaded feature dataset:", df.shape)

    texts = (df['subject'].fillna('') + " " + df['body'].fillna('')).tolist()

    model = load_embedding_model()
    embeddings = generate_embeddings(texts, model)

    embed_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    final_df = pd.concat([df.reset_index(drop=True), embed_df], axis=1)

    final_df.to_csv("data/processed/features_with_embeddings.csv", index=False)
    print("Saved:", final_df.shape)

if __name__ == "__main__":
    add_embeddings_to_dataset()