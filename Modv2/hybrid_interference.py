import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

from script3 import LSTMModel
from script5 import MLPNoisePredictor, betas

device = torch.device('cpu')

# Load models
lstm = LSTMModel().to(device)
lstm.load_state_dict(torch.load('lstm_model.pth'))
lstm.eval()

diffusion = MLPNoisePredictor().to(device)
diffusion.load_state_dict(torch.load('diffusion_model.pth'))
diffusion.eval()

def denoise_step(xt, t):
    with torch.no_grad():
        pred_noise = diffusion(xt, torch.tensor([t]).to(device))
        alpha = 1 - betas[t]
        return (xt - (1 - alpha) / torch.sqrt(1 - torch.cumprod(1 - betas[:t+1], dim=0)) * pred_noise) / torch.sqrt(alpha)

def generate_hybrid(seq, steps=10):
    # LSTM prior
    with torch.no_grad():
        lstm_pred = lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy().flatten()
    
    # Diffusion refinement: Start from noise, denoise to residual
    residual = torch.randn(1, 2).to(device)  # Batch 1, dim 2
    for t in reversed(range(1000)):
        if t % (1000 // steps) == 0:  # Sample fewer steps for speed
            residual = denoise_step(residual, t)
    
    refined = lstm_pred + residual.squeeze().cpu().numpy()
    return refined

# Example: Generate from test seq
data = np.load('sequences.npz')
X_test = data['X_test'][0]  # One seq
hybrid_next = generate_hybrid(X_test)
logging.info(f"Hybrid prediction: {hybrid_next}")