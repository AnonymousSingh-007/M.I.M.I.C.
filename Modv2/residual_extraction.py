import torch
import numpy as np
import logging

# ────────────────────────────────────────────────
# Setup logging (same style as your other scripts)
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

# ────────────────────────────────────────────────
# Define LSTMModel class here (same as in training script)
# This avoids import problems across files
# ────────────────────────────────────────────────
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = torch.nn.Linear(hidden_size, 2)  # predict dx, dy

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(out[:, -1, :])  # last time step

# ────────────────────────────────────────────────
# Main extraction logic
# ────────────────────────────────────────────────
device = torch.device('cpu')

try:
    # Load trained weights
    model = LSTMModel()
    model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
    model.to(device)
    model.eval()
    logging.info("LSTM model loaded successfully from lstm_model.pth")
except Exception as e:
    logging.error(f"Failed to load lstm_model.pth → {e}")
    exit(1)

# Load test sequences
try:
    data = np.load('sequences.npz')
    X_test = data['X_test']
    y_test = data['y_test']
    logging.info(f"Loaded test set: {X_test.shape[0]} samples")
except Exception as e:
    logging.error(f"Failed to load sequences.npz → {e}")
    exit(1)

# Compute residuals
residuals = []
with torch.no_grad():
    for i in range(len(X_test)):
        seq = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(device)
        lstm_pred = model(seq).cpu().numpy().flatten()
        true_next = y_test[i]
        residual = true_next - lstm_pred
        residuals.append(residual)

residuals = np.array(residuals)
np.save('residuals.npy', residuals)

logging.info(f"Successfully extracted and saved {len(residuals):,} residuals → residuals.npy")
logging.info(f"Residual shape: {residuals.shape}")