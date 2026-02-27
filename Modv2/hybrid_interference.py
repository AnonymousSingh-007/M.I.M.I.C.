# hybrid_interference.py  (fixed - no external imports for models)
import torch
import numpy as np
import logging
import random  # for choosing random test sample

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

device = torch.device('cpu')

# ────────────────────────────────────────────────
# Define LSTMModel (same as in training)
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
        self.fc = torch.nn.Linear(hidden_size, 2)  # dx, dy

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        return self.fc(out[:, -1, :])

# ────────────────────────────────────────────────
# Define Diffusion model (same as in diffusion training)
# ────────────────────────────────────────────────
class MLPNoisePredictor(torch.nn.Module):
    def __init__(self, dim=2, hidden=128, embed_dim=32):
        super().__init__()
        self.time_embed = torch.nn.Sequential(
            torch.nn.Linear(1, embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(embed_dim, embed_dim)
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + embed_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, dim)
        )

    def forward(self, x, t):
        t_norm = t.float() / 1000.0
        t_emb = self.time_embed(t_norm.unsqueeze(-1))
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)

# ────────────────────────────────────────────────
# Load both models
# ────────────────────────────────────────────────
try:
    lstm = LSTMModel()
    lstm.load_state_dict(torch.load('lstm_model.pth', map_location=device))
    lstm.to(device)
    lstm.eval()
    logging.info("LSTM model loaded")

    diffusion = MLPNoisePredictor()
    diffusion.load_state_dict(torch.load('diffusion_model.pth', map_location=device))
    diffusion.to(device)
    diffusion.eval()
    logging.info("Diffusion model loaded")
except Exception as e:
    logging.error(f"Model loading failed: {e}")
    exit(1)

# Diffusion schedule (must match training)
num_steps = 1000
betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0).unsqueeze(-1)

def denoise_sample(xt, t):
    with torch.no_grad():
        pred_noise = diffusion(xt, torch.tensor([t], device=device))
        alpha_bar = alphas_cumprod[t]
        alpha = alphas[t]
        # Simplified DDPM reverse step
        mean = (xt - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
        return mean  # no variance added for deterministic sampling

# Hybrid generation function
def generate_hybrid(seq, diffusion_steps=10):
    # seq: (50, 4) numpy array from test set
    seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

    # 1. LSTM deterministic prior
    with torch.no_grad():
        lstm_pred = lstm(seq_tensor).cpu().numpy().flatten()  # shape (2,)

    # 2. Diffusion refinement on residual (start from pure noise)
    residual = torch.randn(1, 2, device=device)  # (1, 2)

    step_interval = num_steps // diffusion_steps
    for step in range(num_steps - 1, -1, -step_interval):
        t = step
        residual = denoise_sample(residual, t)

    refined_residual = residual.squeeze().cpu().numpy()  # (2,)

    # Final hybrid prediction = LSTM + refined residual
    hybrid_pred = lstm_pred + refined_residual

    return hybrid_pred, lstm_pred  # return both for comparison

# ────────────────────────────────────────────────
# Quick test: pick random test sequence and generate
# ────────────────────────────────────────────────
data = np.load('sequences.npz')
X_test = data['X_test']

idx = random.randint(0, len(X_test) - 1)
sample_seq = X_test[idx]

hybrid, lstm_only = generate_hybrid(sample_seq)

logging.info(f"Test sample #{idx}")
logging.info(f"LSTM only prediction (dx, dy): {lstm_only}")
logging.info(f"Hybrid (LSTM + Diffusion) prediction: {hybrid}")
logging.info("Hybrid inference test complete.")