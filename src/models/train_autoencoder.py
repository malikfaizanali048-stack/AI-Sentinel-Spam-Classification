import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train_autoencoder():
    df = pd.read_csv("data/processed/train_final.csv")
    df = df.dropna(subset=['body'])

    ham_df = df[df['label'] == 0]
    print("Ham emails for autoencoder training:", ham_df.shape)

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    X = ham_df[emb_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_val = train_test_split(X_scaled, test_size=0.1, random_state=42)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    model = Autoencoder(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, X_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t)
                val_loss = criterion(val_output, X_val_t)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {loss.item():.4f} - Val Loss: {val_loss.item():.4f}")

    torch.save(model.state_dict(), "data/processed/autoencoder.pt")
    joblib.dump(scaler, "data/processed/autoencoder_scaler.pkl")
    print("Autoencoder saved.")

    return model, scaler

if __name__ == "__main__":
    train_autoencoder()