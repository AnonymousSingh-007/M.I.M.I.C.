import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

device = torch.device('cpu')
residuals = np.load('residuals.npy')
residuals = torch.tensor(residuals, dtype=torch.float32)

class MLPNoisePredictor(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim + 1, hidden)  # +1 for timestep
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, dim)

    def forward(self, x, t):
        t = t.unsqueeze(-1).float() / 1000  # Normalize t
        inp = torch.cat([x, t], dim=-1)
        out = torch.relu(self.fc1(inp))
        out = torch.relu(self.fc2(out))
        return self.fc3(out)

def add_noise(x, t, betas):
    noise = torch.randn_like(x)
    alpha = 1 - betas[t]
    alpha_bar = torch.cumprod(alpha, dim=0)
    return torch.sqrt(alpha_bar) * x + torch.sqrt(1 - alpha_bar) * noise, noise

betas = torch.linspace(1e-4, 0.02, 1000).to(device)  # Diffusion steps

model = MLPNoisePredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loader = DataLoader(TensorDataset(residuals), batch_size=64, shuffle=True)

epochs = 10
for epoch in range(epochs):
    model.train()
    loss_total = 0
    for batch in loader:
        x0 = batch[0].to(device)
        t = torch.randint(0, 1000, (x0.shape[0],)).to(device)
        xt, noise = add_noise(x0, t, betas)
        pred_noise = model(xt, t)
        loss = nn.MSELoss()(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_total += loss.item()
    logging.info(f"Epoch {epoch+1}, Loss: {loss_total/len(loader):.4f}")

torch.save(model.state_dict(), 'diffusion_model.pth')
logging.info("Diffusion training complete")