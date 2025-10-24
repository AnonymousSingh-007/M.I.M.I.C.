# mimic/lstm_trainer.py

# import os
# import joblib
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset
# from mimic.model import LSTMModel


# # ============================
# # Sequence Builder
# # ============================
# def create_sequences(data, seq_len, pred_horizon=2):
#     """
#     Creates (sequence, target) pairs from feature data.
#     - seq = past `seq_len` frames
#     - target = next `pred_horizon` frames (dx, dy)
#     """
#     sequences = []
#     for i in range(len(data) - seq_len - pred_horizon):
#         seq = data[i:i+seq_len]
#         target = data[i+seq_len:i+seq_len+pred_horizon, :2]  # only dx, dy in target
#         sequences.append((seq, target.flatten()))
#     return sequences


# # ============================
# # Training Function
# # ============================
# def train_lstm(
#     csv_path,
#     model_path="models/mimic_lstm.pt",
#     scaler_path="models/mimic_scaler.pkl",
#     seq_len=70,
#     pred_horizon=2,
#     hidden_size=128,
#     num_layers=2,
#     epochs=200,
#     batch_size=128,
#     lr=0.001
# ):
#     """
#     Train LSTM to predict multiple future (dx, dy) steps from past sequence.
#     Normalizes per-session with StandardScaler.
#     Features: dx, dy, speed, dt, curvature
#     """
#     df = pd.read_csv(csv_path)

#     # Use pre-computed features from collector.py
#     features = df[["dx", "dy", "speed", "dt", "curvature"]].values.astype(np.float32)
    
#     # normalize features
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)

#     # build sequences
#     sequences = create_sequences(features_scaled, seq_len, pred_horizon=pred_horizon)
#     if not sequences:
#         raise ValueError("Not enough data to create sequences. Please collect more mouse data.")
        
#     X = np.array([s[0] for s in sequences], dtype=np.float32)
#     y = np.array([s[1] for s in sequences], dtype=np.float32)

#     dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # model
#     model = LSTMModel(input_size=5, hidden_size=hidden_size,
#                       num_layers=num_layers, output_size=pred_horizon*2)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)

#     # training loop
#     for epoch in range(1, epochs + 1):
#         model.train()
#         total_loss = 0.0
#         for X_batch, y_batch in loader:
#             optimizer.zero_grad()
#             output = model(X_batch)
#             loss = criterion(output, y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         if epoch % 10 == 0:
#             avg_loss = total_loss / len(loader)
#             print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.6f}")

#     # save
#     os.makedirs(os.path.dirname(model_path), exist_ok=True)
#     torch.save(model.state_dict(), model_path)
#     joblib.dump(scaler, scaler_path)
#     print("✅ Model and scaler saved.")

#     return model, scaler


# # ============================
# # Load Model + Scaler
# # ============================
# def load_model_and_scaler(model_path="models/mimic_lstm.pt",
#                           scaler_path="models/mimic_scaler.pkl",
#                           pred_horizon=2,
#                           hidden_size=128,
#                           num_layers=2):
#     scaler = joblib.load(scaler_path)
#     model = LSTMModel(input_size=5, hidden_size=hidden_size,
#                       num_layers=num_layers, output_size=pred_horizon*2)
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()
#     return model, scaler


# if __name__ == "__main__":
#     csv_path = "../data/mouse_data.csv"
#     model_path = "../models/mimic_lstm.pt"
#     scaler_path = "../models/mimic_scaler.pkl"

#     train_lstm(
#         csv_path=csv_path,
#         model_path=model_path,
#         scaler_path=scaler_path,
#         seq_len=100,
#         epochs=200,
#         batch_size=128
#     )

# mimic/lstm_trainer.py

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset
# import torch
# import torch.nn as nn
# import torch.optim as optim

# from mimic.model import Seq2SeqModel


# def prepare_sequences(df, seq_len=100, pred_horizon=20):
#     """Prepare overlapping sequences for supervised training."""
#     X, Y = [], []
#     values = df.values
#     for i in range(len(values) - seq_len - pred_horizon):
#         X.append(values[i : i + seq_len])
#         Y.append(values[i + seq_len : i + seq_len + pred_horizon])
#     return np.array(X), np.array(Y)


# def train_model(csv_path, model_path, scaler_path, seq_len=100, pred_horizon=20,
#                 hidden_size=128, num_layers=2, epochs=50, batch_size=64,
#                 lr=1e-3, device="cpu", model_type="seq2seq"):
#     """Train Seq2Seq model on mouse movement data."""
#     print(f"🧠 Training {model_type.upper()} model for {epochs} epochs...")

#     df = pd.read_csv(csv_path)
#     scaler = StandardScaler()
#     data_scaled = scaler.fit_transform(df.values)
#     df_scaled = pd.DataFrame(data_scaled, columns=df.columns)

#     joblib.dump(scaler, scaler_path)

#     X, Y = prepare_sequences(df_scaled, seq_len=seq_len, pred_horizon=pred_horizon)
#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     Y_tensor = torch.tensor(Y, dtype=torch.float32)

#     dataset = TensorDataset(X_tensor, Y_tensor)
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     model = Seq2SeqModel(
#         input_size=X.shape[2], hidden_size=hidden_size, num_layers=num_layers, output_size=Y.shape[2]
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.MSELoss()

#     for epoch in range(1, epochs + 1):
#         model.train()
#         total_loss = 0
#         for X_batch, Y_batch in loader:
#             X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

#             optimizer.zero_grad()
#             outputs = model(X_batch, target_len=pred_horizon)
#             loss = criterion(outputs, Y_batch)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         avg_loss = total_loss / len(loader)
#         if epoch % 10 == 0 or epoch == 1:
#             print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

#     torch.save(model.state_dict(), model_path)
#     print(f"✔ Model saved to {model_path}")


#gemini approach

# mimic/lstm_trainer.py

# # mimic/lstm_trainer.py

# import os
# import joblib
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from rich.console import Console
# from rich.progress import Progress
# from mimic.model import Seq2SeqModel

# console = Console()


# def prepare_sequences(df, seq_len=100, pred_horizon=20):
#     data = df.to_numpy(dtype=np.float32)
#     if len(data) < seq_len + pred_horizon:
#         console.print("[yellow]Not enough data for sequences.[/yellow]")
#         return None, None

#     n_features = data.shape[1]
#     n_samples = len(data) - seq_len - pred_horizon + 1

#     X = np.lib.stride_tricks.as_strided(
#         data,
#         shape=(n_samples, seq_len, n_features),
#         strides=(data.strides[0], data.strides[0], data.strides[1])
#     )
#     Y = np.lib.stride_tricks.as_strided(
#         data[seq_len:],
#         shape=(n_samples, pred_horizon, n_features),
#         strides=(data.strides[0], data.strides[0], data.strides[1])
#     )
#     return X, Y


# def train_model(csv_path, model_path, scaler_path,
#                 seq_len=100, pred_horizon=20, hidden_size=128, num_layers=2,
#                 epochs=50, batch_size=64, lr=1e-3, model_type="seq2seq"):
#     console.print(f"🧠 Training {model_type.upper()}...")
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     if not os.path.exists(csv_path):
#         console.print("[red]Data file missing.[/red]")
#         return

#     df = pd.read_csv(csv_path)
#     scaler = StandardScaler()
#     scaled = scaler.fit_transform(df.values)
#     joblib.dump(scaler, scaler_path)
#     console.print(f"✔ Scaler saved → {scaler_path}")

#     X, Y = prepare_sequences(pd.DataFrame(scaled, columns=df.columns),
#                              seq_len=seq_len, pred_horizon=pred_horizon)
#     if X is None:
#         return

#     dataset = TensorDataset(torch.tensor(X), torch.tensor(Y))
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     model = Seq2SeqModel(X.shape[2], hidden_size, num_layers, X.shape[2]).to(device)
#     opt = optim.Adam(model.parameters(), lr=lr)
#     loss_fn = nn.MSELoss()

#     with Progress() as prog:
#         task = prog.add_task("Training...", total=epochs)
#         for ep in range(1, epochs + 1):
#             model.train()
#             total_loss = 0
#             for xb, yb in loader:
#                 xb, yb = xb.to(device), yb.to(device)
#                 opt.zero_grad()
#                 out = model(xb, trg=yb)
#                 loss = loss_fn(out, yb)
#                 loss.backward()
#                 opt.step()
#                 total_loss += loss.item()
#             avg = total_loss / len(loader)
#             prog.update(task, advance=1, description=f"Epoch {ep}/{epochs} | Loss {avg:.4f}")

#     torch.save(model.state_dict(), model_path)
#     console.print(f"✔ Model saved → {model_path}")









#mimic/lstm_trainer.py - E




# mimic/lstm_trainer.py
# mimic/lstm_trainer.py
# mimic/lstm_trainer.py
import os
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from mimic.model import LSTMModel

def create_sequences(data, seq_len, horizon=2):
    sequences = []
    for i in range(len(data) - seq_len - horizon):
        seq = data[i:i+seq_len]
        target = data[i+seq_len:i+seq_len+horizon, :2].flatten()
        sequences.append((seq, target))
    return sequences

def train_lstm(csv_path,
               model_path="models/mimic_lstm.pt",
               scaler_path="models/mimic_scaler.pkl",
               seq_len=150, horizon=2,
               hidden_size=256, num_layers=2,
               epochs=200, batch_size=256, lr=0.001):

    df = pd.read_csv(csv_path)
    features = df[["dx", "dy", "speed", "dt"]].values.astype(np.float32)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    sequences = create_sequences(features_scaled, seq_len, horizon)
    if not sequences:
        raise ValueError("Not enough data to create sequences.")
    X = np.array([s[0] for s in sequences], dtype=np.float32)
    y = np.array([s[1] for s in sequences], dtype=np.float32)

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(input_size=4, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=horizon*2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {total_loss/len(loader):.6f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    print("✅ Model and scaler saved.")
    return model, scaler

def load_model_and_scaler(model_path="models/mimic_lstm.pt",
                          scaler_path="models/mimic_scaler.pkl",
                          horizon=2, hidden_size=256, num_layers=2):
    scaler = joblib.load(scaler_path)
    model = LSTMModel(input_size=4, hidden_size=hidden_size,
                      num_layers=num_layers, output_size=horizon*2)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model, scaler
