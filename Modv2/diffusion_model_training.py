import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)

device = torch.device('cpu')

# Load residuals (should be (N, 2))
residuals = np.load('residuals.npy')
residuals = torch.tensor(residuals, dtype=torch.float32).to(device)
logging.info(f"Loaded residuals: shape {residuals.shape}")

# Simple MLP noise predictor (lightweight, as per PDF)
class MLPNoisePredictor(nn.Module):
    def __init__(self, dim=2, hidden=128, embed_dim=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(dim + embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim)
        )

    def forward(self, x, t):
        # t: (batch,) → normalize & embed
        t_norm = t.float() / 1000.0
        t_emb = self.time_embed(t_norm.unsqueeze(-1))
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)

# Diffusion schedule (standard linear)
num_steps = 1000
betas = torch.linspace(1e-4, 0.02, num_steps).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)          # shape (1000,)
alphas_cumprod = alphas_cumprod.unsqueeze(-1)          # shape (1000, 1) for broadcasting

def add_noise(x0, t):
    """
    q(x_t | x_0) = sqrt(α_bar_t) x_0 + sqrt(1 - α_bar_t) ε
    """
    noise = torch.randn_like(x0)
    alpha_bar_t = alphas_cumprod[t]                    # (batch, 1)
    mean = torch.sqrt(alpha_bar_t) * x0
    std = torch.sqrt(1 - alpha_bar_t)
    xt = mean + std * noise
    return xt, noise

# Model, optimizer, loader
model = MLPNoisePredictor(dim=2, hidden=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
dataset = TensorDataset(residuals)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)  # num_workers=0 safe on Windows

epochs = 20  # start here; increase to 50+ later if needed
model.train()

for epoch in range(epochs):
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x0 = batch[0].to(device)                   # (batch, 2)
        batch_size = x0.shape[0]

        t = torch.randint(0, num_steps, (batch_size,), device=device)  # (batch,)

        xt, noise = add_noise(x0, t)

        pred_noise = model(xt, t)
        loss = nn.MSELoss()(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    logging.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), 'diffusion_model.pth')
logging.info("Diffusion training complete. Saved diffusion_model.pth")