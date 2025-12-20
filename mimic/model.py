import torch.nn as nn


class ImprovedLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=512, horizon=10):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=3,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, horizon * 2)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])
