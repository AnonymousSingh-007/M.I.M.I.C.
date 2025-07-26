import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from pathlib import Path

# 🔧 Model definition
class MouseMLP(nn.Module):
    def __init__(self):
        super(MouseMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # output: x, y
        )

    def forward(self, x):
        return self.net(x)


# 🧠 Training function
def train(csv_path: str, model_path: str = "models/mimic_model.pt", epochs: int = 200):
    df = pd.read_csv(csv_path)

    # --- FIX APPLIED HERE ---
    timestamps = df['timestamp'].to_numpy().reshape(-1, 1)
    positions = df[['x', 'y']].to_numpy() # Using .to_numpy() is the recommended modern approach

    # Normalize time to improve learning
    timestamps = timestamps - timestamps.min()
    timestamps = torch.tensor(timestamps, dtype=torch.float32)
    positions = torch.tensor(positions, dtype=torch.float32)

    dataset = TensorDataset(timestamps, positions)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MouseMLP()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    # Save the model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")


# 📦 Load model for inference
def load_model(model_path: str = "models/mimic_model.pt"):
    model = MouseMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# 🔮 Prediction function
def predict(model, timestamp: float, t0: float):
    relative_time = torch.tensor([[timestamp - t0]], dtype=torch.float32)
    with torch.no_grad():
        xy = model(relative_time).squeeze().tolist()
    return xy