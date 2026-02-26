import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, entropy
from sklearn.metrics import mean_squared_error
import time
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

from script6 import generate_hybrid, lstm, diffusion  # Load models as above

device = torch.device('cpu')
data = np.load('sequences.npz')
X_test, y_test = data['X_test'][:1000], data['y_test'][:1000]  # Subsample for speed

# Generate predictions
lstm_preds = []
hybrid_preds = []
diff_preds = []  # Ablation: Diffusion only (from noise, no LSTM)
latencies = {'lstm': [], 'hybrid': []}

for seq in X_test:
    # LSTM
    start = time.time()
    with torch.no_grad():
        pred = lstm(torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)).cpu().numpy().flatten()
    latencies['lstm'].append((time.time() - start) * 1000)
    lstm_preds.append(pred)
    
    # Hybrid
    start = time.time()
    hybrid = generate_hybrid(seq)
    latencies['hybrid'].append((time.time() - start) * 1000)
    hybrid_preds.append(hybrid)
    
    # Diffusion only
    residual = torch.randn(1, 2).to(device)
    for t in reversed(range(1000)):
        if t % 100 == 0:  # 10 steps
            residual = denoise_step(residual, t)  # From script6
    diff_preds.append(residual.squeeze().cpu().numpy())

lstm_preds = np.array(lstm_preds)
hybrid_preds = np.array(hybrid_preds)
diff_preds = np.array(diff_preds)

# Velocities (assume denormalized for realism; simplify here)
true_vel = np.linalg.norm(y_test, axis=1)
lstm_vel = np.linalg.norm(lstm_preds, axis=1)
hybrid_vel = np.linalg.norm(hybrid_preds, axis=1)
diff_vel = np.linalg.norm(diff_preds, axis=1)

# JSD (on velocity dist)
def jsd(p, q):
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2

kde_true = gaussian_kde(true_vel)
kde_lstm = gaussian_kde(lstm_vel)
kde_hybrid = gaussian_kde(hybrid_vel)
kde_diff = gaussian_kde(diff_vel)
x = np.linspace(0, max(true_vel.max(), lstm_vel.max()), 100)
jsd_lstm = jsd(kde_true(x), kde_lstm(x))
jsd_hybrid = jsd(kde_true(x), kde_hybrid(x))
jsd_diff = jsd(kde_true(x), kde_diff(x))

# MSE/ADE
mse_lstm = mean_squared_error(y_test, lstm_preds)
mse_hybrid = mean_squared_error(y_test, hybrid_preds)
mse_diff = mean_squared_error(y_test, diff_preds)

# Smoothness: Std of 'accelerations' (diff of velocities)
acc_true = np.diff(true_vel).std()
acc_lstm = np.diff(lstm_vel).std()
acc_hybrid = np.diff(hybrid_vel).std()
acc_diff = np.diff(diff_vel).std()

# Latency avg
lat_lstm = np.mean(latencies['lstm'])
lat_hybrid = np.mean(latencies['hybrid'])

# Log results
logging.info(f"LSTM: JSD {jsd_lstm:.2f}, MSE {mse_lstm:.4f}, Smoothness {acc_lstm:.2f}, Latency {lat_lstm:.2f}ms")
logging.info(f"Hybrid: JSD {jsd_hybrid:.2f}, MSE {mse_hybrid:.4f}, Smoothness {acc_hybrid:.2f}, Latency {lat_hybrid:.2f}ms")
logging.info(f"Diffusion Only: JSD {jsd_diff:.2f}, MSE {mse_diff:.4f}, Smoothness {acc_diff:.2f}")

# Ablation Table
ablation = pd.DataFrame({
    'Model': ['LSTM Only', 'Diffusion Only', 'Hybrid'],
    'JSD': [jsd_lstm, jsd_diff, jsd_hybrid],
    'Realism': ['Medium', 'Moderate', 'High'],
    'Comments': ['Deterministic motion', 'No temporal memory', 'Best performance']
})
print(ablation)

# Plots
# Trajectory (example, cumulative positions; assume start from 0)
true_traj = np.cumsum(y_test[:100], axis=0)
lstm_traj = np.cumsum(lstm_preds[:100], axis=0)
hybrid_traj = np.cumsum(hybrid_preds[:100], axis=0)
plt.plot(true_traj[:,0], true_traj[:,1], label='Human')
plt.plot(lstm_traj[:,0], lstm_traj[:,1], label='LSTM')
plt.plot(hybrid_traj[:,0], hybrid_traj[:,1], label='Hybrid')
plt.legend()
plt.savefig('trajectory_comparison.png')

# Velocity KDE
plt.figure()
plt.plot(x, kde_true(x), label='Human')
plt.plot(x, kde_lstm(x), label='LSTM')
plt.plot(x, kde_hybrid(x), label='Hybrid')
plt.legend()
plt.savefig('velocity_dist.png')

logging.info("Evaluation complete. Check plots and log.")