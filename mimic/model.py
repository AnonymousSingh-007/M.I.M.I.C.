#mimic/model.py
#curvature
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    input_size = number of features (dx, dy, speed, dt, curvature)
    output_size = horizon * 2 (predict dx, dy for horizon steps)
    """
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=8):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
