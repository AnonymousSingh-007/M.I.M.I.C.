import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()])

from script3 import LSTMModel  # Import from above

device = torch.device('cpu')
model = LSTMModel()
model.load_state_dict(torch.load('lstm_model.pth'))
model.to(device)
model.eval()

data = np.load('sequences.npz')
X_test, y_test = data['X_test'], data['y_test']  # y_test is true next dx,dy

residuals = []
with torch.no_grad():
    for i in range(len(X_test)):
        seq = torch.tensor(X_test[i], dtype=torch.float32).unsqueeze(0).to(device)
        lstm_pred = model(seq).cpu().numpy().flatten()
        residual = y_test[i] - lstm_pred
        residuals.append(residual)

residuals = np.array(residuals)
np.save('residuals.npy', residuals)
logging.info(f"Extracted {len(residuals)} residuals")