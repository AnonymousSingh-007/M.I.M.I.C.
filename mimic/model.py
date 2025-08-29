# #trying better pipeline integration
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from pathlib import Path
# import os
# from typing import Optional

# MODEL_PATH = "models/mimic_model.pt"


# # 🔧 Model Definition
# class MouseMLP(nn.Module):
#     def __init__(self, input_dim=5, hidden_dim=64):
#         super(MouseMLP, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2)  # Output: x, y
#         )

#     def forward(self, x):
#         return self.model(x)


# # 🧠 Train Function with Auto-Preprocessing
# def train(csv_path: str, model_path: str = MODEL_PATH, epochs: int = 100) -> Optional[MouseMLP]:
#     if not os.path.exists(csv_path):
#         print(f"[ERROR] File not found: {csv_path}")
#         return None

#     df = pd.read_csv(csv_path)

#     # Auto-generate required columns if only raw timestamp/x/y exist
#     if set(df.columns) == {"timestamp", "x", "y"}:
#         df = df.sort_values("timestamp").reset_index(drop=True)
#         df["time"] = df["timestamp"] - df["timestamp"].iloc[0]
#         df["dx"] = df["x"].diff().fillna(0)
#         df["dy"] = df["y"].diff().fillna(0)
#         df["speed"] = (df["dx"] ** 2 + df["dy"] ** 2) ** 0.5
#         df["pause"] = (df["time"].diff().fillna(0).astype(float) > 0.05).astype(int)
#         df = df[["time", "dx", "dy", "speed", "pause", "x", "y"]]

#     required = ['time', 'dx', 'dy', 'speed', 'pause', 'x', 'y']
#     if not all(col in df.columns for col in required):
#         print(f"[ERROR] CSV missing required columns: {', '.join(required)}")
#         return None

#     features = df[['time', 'dx', 'dy', 'speed', 'pause']].to_numpy(dtype='float32')
#     targets = df[['x', 'y']].to_numpy(dtype='float32')

#     # Normalize features
#     features_mean = features.mean(axis=0)
#     features_std = features.std(axis=0)
#     features = (features - features_mean) / features_std

#     X = torch.tensor(features, dtype=torch.float32)
#     y = torch.tensor(targets, dtype=torch.float32)

#     dataset = torch.utils.data.TensorDataset(X, y)
#     loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

#     model_net = MouseMLP(input_dim=5)
#     optimizer = optim.Adam(model_net.parameters(), lr=0.001)
#     loss_fn = nn.MSELoss()

#     for epoch in range(epochs):
#         model_net.train()
#         total_loss = 0.0
#         for x_batch, y_batch in loader:
#             optimizer.zero_grad()
#             y_pred = model_net(x_batch)
#             loss = loss_fn(y_pred, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         if epoch % 10 == 0 or epoch == epochs - 1:
#             print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}")

#     Path(model_path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save({
#         'model_state': model_net.state_dict(),
#         'mean': features_mean,
#         'std': features_std
#     }, model_path)

#     print(f"✅ Model trained and saved to {model_path}")
#     return model_net


# # 📦 Load Model with Normalization
# def load_model(model_path: str = MODEL_PATH) -> MouseMLP:
#     checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
#     model_net = MouseMLP(input_dim=5)
#     model_net.load_state_dict(checkpoint['model_state'])
#     model_net.eval()
#     model_net.mean = checkpoint['mean']
#     model_net.std = checkpoint['std']
#     return model_net


# # 🔮 Predict Next Position from Input Vector
# def predict(model: MouseMLP, rel_time, dx, dy, speed, pause):
#     inputs = torch.tensor([[rel_time, dx, dy, speed, pause]], dtype=torch.float32)
#     inputs = (inputs - torch.tensor(model.mean)) / torch.tensor(model.std)
#     with torch.no_grad():
#         xy = model(inputs).squeeze().tolist()
#     return xy

#trying lstm now
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path

class MouseLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, output_size=2, num_layers=2):
        super(MouseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(data, seq_len):
    sequences, targets = [], []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
        targets.append(data[i+seq_len])
    return np.array(sequences), np.array(targets)


def train_lstm(csv_path: str, model_path="models/mimic_lstm.pt", epochs=150, seq_len=30):
    df = pd.read_csv(csv_path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("CSV must contain 'x' and 'y' columns")

    data = df[["x", "y"]].to_numpy(dtype=np.float32)

    # scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # save scaler
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # sequences
    X, y = create_sequences(data_scaled, seq_len)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    model = MouseLSTM()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), model_path)
    print(f"LSTM model trained and saved to {model_path}")
    return model


def load_model_and_scaler(model_path: str = "models/mimic_lstm.pt"):
    """Load trained LSTM model + scaler"""
    model = MouseLSTM(input_size=2, hidden_size=64, output_size=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    scaler = joblib.load("models/scaler.pkl")
    return model, scaler
