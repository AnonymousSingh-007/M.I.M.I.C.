import torch
import pandas as pd
import numpy as np
import joblib

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from mimic.model import ImprovedLSTM


def train_lstm_fixed(
    csv_file,
    model_path,
    scaler_path,
    seq_len=200,
    horizon=10,
    epochs=80,
    batch_size=256
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CSV...")
    df = pd.read_csv(csv_file)

    print("Preparing features...")
    features = df[["x", "y", "dx", "dy", "speed", "dt"]].values.astype(np.float32)

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    n_samples = len(features) - seq_len - horizon
    if n_samples <= 0:
        raise ValueError("Not enough data for given seq_len/horizon")

    print(f"Building {n_samples:,} sequences (vectorized)...")

    # Vectorized sequence construction
    X = np.lib.stride_tricks.sliding_window_view(
        features, (seq_len, features.shape[1])
    )[:, 0][:n_samples]

    y = np.empty((n_samples, horizon * 2), dtype=np.float32)
    for i in range(n_samples):
        y[i] = features[i + seq_len:i + seq_len + horizon, 2:4].flatten()

    print("Converting to tensors...")
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).to(device)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=True
    )

    model = ImprovedLSTM(horizon=horizon).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"Epoch {epoch:03d}/{epochs} | Loss: {avg:.6f}")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print("Model and scaler saved")
