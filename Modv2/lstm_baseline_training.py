# lstm_baseline_training.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import time

# Setup logging (same as preprocessing)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

device = torch.device('cpu')  # No GPU → force CPU

# Load sequences
try:
    data = np.load('sequences.npz')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val   = data['X_val']
    y_val   = data['y_val']
    logging.info(f"Loaded sequences: Train {X_train.shape}, Val {X_val.shape}")
except Exception as e:
    logging.error(f"Failed to load sequences.npz: {e}")
    exit(1)

# Datasets & Loaders (small batch for 8GB RAM)
batch_size = 64
train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

# LSTM Model (small & lightweight)
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0  # no dropout for tiny model
        )
        self.fc = nn.Linear(hidden_size, 2)  # output: dx, dy

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(out[:, -1, :])  # take last timestep

model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 15  # start small — increase to 30–50 later if needed
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    start_time = time.time()

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_x.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            val_loss += criterion(pred, batch_y).item() * batch_x.size(0)

    val_loss /= len(val_loader.dataset)

    epoch_time = time.time() - start_time
    logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {epoch_time:.1f}s")

    # Simple save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'lstm_model_best.pth')
        logging.info(f"→ Saved better model (val loss: {val_loss:.6f})")

# Final save
torch.save(model.state_dict(), 'lstm_model.pth')
logging.info("LSTM training finished. Model saved as lstm_model.pth and lstm_model_best.pth")