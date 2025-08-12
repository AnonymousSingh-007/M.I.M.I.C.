# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import TensorDataset, DataLoader
# from pathlib import Path
# import pandas as pd
# import os

# # 🔧 Model definition
# class MouseMLP(nn.Module):
#     def __init__(self):
#         super(MouseMLP, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(1, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 2)  # output: x, y
#         )

#     def forward(self, x):
#         return self.net(x)


# # 🧠 Training function
# def train(csv_path: str, model_path: str = "models/mimic_model.pt", epochs: int = 200):
#     df = pd.read_csv(csv_path)

#     # --- FIX APPLIED HERE ---
#     timestamps = df['timestamp'].to_numpy().reshape(-1, 1)
#     positions = df[['x', 'y']].to_numpy() # Using .to_numpy() is the recommended modern approach

#     # Normalize time to improve learning
#     timestamps = timestamps - timestamps.min()
#     timestamps = torch.tensor(timestamps, dtype=torch.float32)
#     positions = torch.tensor(positions, dtype=torch.float32)

#     dataset = TensorDataset(timestamps, positions)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

#     model = MouseMLP()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     loss_fn = nn.MSELoss()

#     for epoch in range(epochs):
#         total_loss = 0
#         for x_batch, y_batch in dataloader:
#             optimizer.zero_grad()
#             y_pred = model(x_batch)
#             loss = loss_fn(y_pred, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
#             print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

#     # Save the model
#     Path(model_path).parent.mkdir(parents=True, exist_ok=True)
#     torch.save(model.state_dict(), model_path)
#     print(f"✅ Model saved to {model_path}")



# # 📦 Load model for inference
# def load_model(model_path: str = "models/mimic_model.pt"):
#     model = MouseMLP()
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model


# # 🔮 Prediction function
# def predict(model, timestamp: float, t0: float):
#     relative_time = torch.tensor([[timestamp - t0]], dtype=torch.float32)
#     with torch.no_grad():
#         xy = model(relative_time).squeeze().tolist()
#     return xy

#Trying Dynamic and idle integration
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from pathlib import Path
import os

MODEL_PATH = "models/mimic_model.pt"


# 🔧 Model Definition
class MouseMLP(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(MouseMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: x, y
        )

    def forward(self, x):
        return self.model(x)


# 🧠 Train Function with Auto-Preprocessing
def train(csv_path: str, model_path: str = MODEL_PATH, epochs: int = 100):
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Auto-generate required columns if only raw timestamp/x/y exist
    if set(df.columns) == {"timestamp", "x", "y"}:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["time"] = df["timestamp"] - df["timestamp"].iloc[0]
        df["dx"] = df["x"].diff().fillna(0)
        df["dy"] = df["y"].diff().fillna(0)
        df["speed"] = (df["dx"] ** 2 + df["dy"] ** 2) ** 0.5
        df["pause"] = (df["time"].diff().fillna(0).astype(float) > 0.05).astype(int)
        df = df[["time", "dx", "dy", "speed", "pause", "x", "y"]]

    required = ['time', 'dx', 'dy', 'speed', 'pause', 'x', 'y']
    if not all(col in df.columns for col in required):
        print(f"[ERROR] CSV missing required columns: {', '.join(required)}")
        return None

    features = df[['time', 'dx', 'dy', 'speed', 'pause']].to_numpy(dtype='float32')
    targets = df[['x', 'y']].to_numpy(dtype='float32')

    # Normalize features
    features_mean = features.mean(axis=0)
    features_std = features.std(axis=0)
    features = (features - features_mean) / features_std

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model_net = MouseMLP(input_dim=5)
    optimizer = optim.Adam(model_net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model_net.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model_net(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.6f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'model_state': model_net.state_dict(),
        'mean': features_mean,
        'std': features_std
    }, model_path)

    print(f"✅ Model trained and saved to {model_path}")
    return model_net


# 📦 Load Model with Normalization
def load_model(model_path: str = MODEL_PATH):
    checkpoint = torch.load(model_path, weights_only=False)
    model_net = MouseMLP(input_dim=5)
    model_net.load_state_dict(checkpoint['model_state'])
    model_net.eval()
    model_net.mean = checkpoint['mean']
    model_net.std = checkpoint['std']
    return model_net


# 🔮 Predict Next Position from Input Vector
def predict(model, rel_time, dx, dy, speed, pause):
    inputs = torch.tensor([[rel_time, dx, dy, speed, pause]], dtype=torch.float32)
    inputs = (inputs - torch.tensor(model.mean)) / torch.tensor(model.std)
    with torch.no_grad():
        xy = model(inputs).squeeze().tolist()
    return xy
