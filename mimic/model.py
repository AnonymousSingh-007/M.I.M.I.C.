
#corrected lSTM
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # predict next dx, dy
        return out


def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):  # ✅ fixed: removed extra ")"
        seq = data[i:i+seq_len]
        target = data[i+seq_len]
        sequences.append((seq, target))
    return sequences


from torch.utils.data import DataLoader, TensorDataset

def train_lstm(csv_path, model_path="models/mimic_lstm.pt",
               scaler_path="models/mimic_scaler.pkl",
               epochs=300, seq_len=100, batch_size=128):
    """
    Train LSTM on recorded mouse movement data (x,y) -> deltas (dx,dy).
    Uses mini-batches to prevent OOM errors.
    """
    df = pd.read_csv(csv_path)

    if not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"CSV must contain 'x' and 'y' columns, got {df.columns.tolist()}")

    # compute deltas
    data = df[["x", "y"]].values.astype(np.float32)
    deltas = np.diff(data, axis=0)

    # scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(deltas)

    # create sequences
    sequences = create_sequences(data_scaled, seq_len)
    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.float32)

    # torch tensors
    X_tensor = torch.tensor(X)
    y_tensor = torch.tensor(y)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print("✅ Model and scaler saved.")


def load_model_and_scaler(model_path="models/mimic_lstm.pt",
                          scaler_path="models/mimic_scaler.pkl"):
    """
    Load trained LSTMModel and scaler for spoofing.
    """
    scaler = joblib.load(scaler_path)

    model = LSTMModel(input_size=2, hidden_size=128, output_size=2, num_layers=2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, scaler
