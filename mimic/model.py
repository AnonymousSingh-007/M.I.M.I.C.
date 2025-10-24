#mimic/model.py
#curvature
# import torch.nn as nn

# class LSTMModel(nn.Module):
#     """
#     input_size = number of features (dx, dy, speed, dt, curvature)
#     output_size = pred_horizon * 2 (predict dx, dy for pred_horizon steps)
#     """
#     def __init__(self, input_size=5, hidden_size=128, num_layers=2, output_size=8):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])
#         return out

# mimic/model.py

# import torch
# import torch.nn as nn
# import pandas as pd


# class Seq2SeqModel(nn.Module):
#     def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=7):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)

#     def forward(self, src, target_len=20):
#         """Teacher-forced forward pass for training."""
#         batch_size = src.size(0)

#         # Encode
#         _, (h, c) = self.encoder(src)

#         # Decoder starts with zeros
#         decoder_input = torch.zeros(batch_size, 1, 7, device=src.device)
#         outputs = []

#         for _ in range(target_len):
#             dec_out, (h, c) = self.decoder(decoder_input, (h, c))
#             step_out = self.fc(dec_out)
#             outputs.append(step_out)
#             decoder_input = step_out  # feeding back output

#         return torch.cat(outputs, dim=1)

#     def generate(self, seed_seq, scaler, pred_horizon=20, device="cpu"):
#         """Autoregressive generation from a seed sequence."""
#         self.eval()
#         with torch.no_grad():
#             df_seed = pd.DataFrame(
#                 seed_seq,
#                 columns=["x", "y", "dx", "dy", "dt", "speed", "accel"],
#             )
#             seed_scaled = scaler.transform(df_seed.values)

#             src = torch.tensor(seed_scaled, dtype=torch.float32, device=device).unsqueeze(0)

#             # Encode
#             _, (h, c) = self.encoder(src)

#             # Start decoder from last input step
#             decoder_input = src[:, -1:, :]  # shape (1, 1, 7)
#             outputs = []

#             for _ in range(pred_horizon):
#                 dec_out, (h, c) = self.decoder(decoder_input, (h, c))
#                 step_out = self.fc(dec_out)
#                 outputs.append(step_out)
#                 decoder_input = step_out  # autoregressive

#             outputs = torch.cat(outputs, dim=1).squeeze(0).cpu().numpy()
#             outputs_unscaled = scaler.inverse_transform(outputs)

#         return outputs_unscaled


#gemini approach

# mimic/model.py

# # mimic/model.py

# import torch
# import torch.nn as nn

# class Seq2SeqModel(nn.Module):
#     def __init__(self, input_size, hidden_size=128, num_layers=2, output_size=None):
#         super().__init__()
#         self.output_size = output_size or input_size
#         self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.decoder = nn.LSTM(self.output_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, self.output_size)
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"

#     def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
#         _, (h, c) = self.encoder(src)
#         dec_in = torch.zeros(src.size(0), 1, self.output_size, device=self.device)
#         outputs = []
#         T = trg.size(1) if trg is not None else 1
#         for t in range(T):
#             dec_out, (h, c) = self.decoder(dec_in, (h, c))
#             pred = self.fc(dec_out)
#             outputs.append(pred)
#             dec_in = trg[:, t:t+1, :] if (trg is not None and torch.rand(1).item() < teacher_forcing_ratio) else pred.detach()
#         return torch.cat(outputs, 1)

#     def generate(self, seed_seq, scaler, horizon=20):
#         self.eval()
#         with torch.no_grad():
#             src = torch.tensor(scaler.transform(seed_seq), dtype=torch.float32, device=self.device).unsqueeze(0)
#             _, (h, c) = self.encoder(src)
#             dec_in, outs = src[:, -1:, :], []
#             for _ in range(horizon):
#                 dec_out, (h, c) = self.decoder(dec_in, (h, c))
#                 step = self.fc(dec_out)
#                 outs.append(step)
#                 dec_in = step
#             out = torch.cat(outs, 1).squeeze(0).cpu().detach().numpy()
#             return scaler.inverse_transform(out)











#mimic/model.py - E




# mimic/model.py
# mimic/model.py
# mimic/model.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=256, num_layers=2, output_size=6):
        """
        input_size: dx, dy, speed, dt
        output_size: horizon*2 (dx, dy for next steps)
        """
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
