# mimic/lstm_trainer.py

import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from mimic.model import LSTMModel

# ============================
# Sequence Builder
# ============================
def create_sequences(data, seq_len, horizon=5):
    """
    Creates (sequence, target) pairs from feature data.
    - seq = past `seq_len` frames
    - target = next `horizon` frames (dx, dy)
    """
    sequences = []
    for i in range(len(data) - seq_len - horizon):
        seq = data[i:i+seq_len]
        target = data[i+seq_len:i+seq_len+horizon, :2]  # only dx, dy in target
        sequences.append((seq, target.flatten()))
    return sequences


# ============================
# Training Function
# ============================
def train_lstm(
    csv_path,
    model_path="models/mimic_lstm.pt",
    scaler_path="models/mimic_scaler.pkl",
    seq_len=70,
    horizon=3,
    hidden_size=128,
    num_layers=2,
    epochs=200,
    batch_size=128,
    lr=0.001
):
    """
    Train LSTM to predict multiple future (dx, dy) steps from past sequence.
    Normalizes per-session with StandardScaler.
    """
    df = pd.read_csv(csv_path)

    # Use pre-computed features from collector.py
    features = df[["dx", "dy", "speed", "dt"]].values.astype(np.float32)
    
    # normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # build sequences
    sequences = create_sequences(features_scaled, seq_len, horizon=horizon)
    if not sequences:
        raise ValueError("Not enough data to create sequences. Please collect more mouse data.")
        
    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.float32)

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = LSTMModel(input_size=4, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=horizon*2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

    # save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print("✅ Model and scaler saved.")

    return model, scaler


# ============================
# Load Model + Scaler
# ============================
def load_model_and_scaler(model_path="models/mimic_lstm.pt",
                          scaler_path="models/mimic_scaler.pkl",
                          horizon=3,
                          hidden_size=128,
                          num_layers=2):
    scaler = joblib.load(scaler_path)
    model = LSTMModel(input_size=4, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=horizon*2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, scaler

if __name__ == "__main__":
    csv_path = "../data/mouse_data.csv"
    model_path = "../models/mimic_lstm.pt"
    scaler_path = "../models/mimic_scaler.pkl"

    train_lstm(
        csv_path=csv_path,
        model_path=model_path,
        scaler_path=scaler_path,
        seq_len=100,
        epochs=250,
        batch_size=128
    )
