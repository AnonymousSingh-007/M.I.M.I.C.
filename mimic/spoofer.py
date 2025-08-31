
# mimic/spoofer.py

import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
import pyautogui
from pynput import mouse

# =============================
#   LSTM Model
# =============================
class MouseLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, output_size=2):
        super(MouseLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =============================
#   Data Collection
# =============================
def collect_data(csv_path="data/mouse_data.csv", duration=30):
    coords = []

    def on_move(x, y):
        coords.append((time.time(), x, y))

    listener = mouse.Listener(on_move=on_move)
    listener.start()

    print(f"📡 Recording mouse movement for {duration} seconds...")
    time.sleep(duration)
    listener.stop()

    df = pd.DataFrame(coords, columns=["time", "x", "y"])
    df["dx"] = df["x"].diff().fillna(0)
    df["dy"] = df["y"].diff().fillna(0)
    df.to_csv(csv_path, index=False)

    print(f"✅ Data saved to {csv_path}")


# =============================
#   Training
# =============================
def train_lstm(csv_path, model_path="models/mimic_lstm.pt",
               scaler_path="models/mimic_scaler.pkl",
               seq_len=100, epochs=50, lr=0.001):

    df = pd.read_csv(csv_path)
    if {"dx", "dy"}.issubset(df.columns):
        data = df[["dx", "dy"]].values
    else:
        coords = df[["x", "y"]].values
        data = np.diff(coords, axis=0)

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - seq_len):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len])
    X, y = np.array(X), np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model = MouseLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    print("✅ Model and scaler saved.")


def load_model_and_scaler(model_path, scaler_path):
    model = MouseLSTM()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler


# =============================
#   Spoofing + Graph
# =============================
def spoof_and_plot(model_path="models/mimic_lstm.pt",
                   scaler_path="models/mimic_scaler.pkl",
                   csv_path="data/mouse_data.csv",
                   seq_len=100,
                   steps=500,
                   move_delay=0.001):

    model, scaler = load_model_and_scaler(model_path, scaler_path)

    df = pd.read_csv(csv_path)
    if {"dx", "dy"}.issubset(df.columns):
        data = df[["dx", "dy"]].values
    else:
        coords = df[["x", "y"]].values
        data = np.diff(coords, axis=0)

    data_scaled = scaler.transform(data)
    seq = torch.tensor(data_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0)

    generated = []
    current_x, current_y = pyautogui.position()

    print("🎮 Spoofing mouse now...")
    with torch.no_grad():
        for _ in range(steps):
            pred = model(seq).numpy()[0]
            dx, dy = scaler.inverse_transform([pred])[0]
            current_x += dx
            current_y += dy
            pyautogui.moveTo(current_x, current_y)   # 👈 move cursor
            time.sleep(move_delay)
            generated.append([dx, dy])
            seq = torch.cat([seq[:, 1:, :], torch.tensor(pred).view(1, 1, -1)], dim=1)

    print("✅ Spoofing finished. Showing graph...")

    # === Graph ===
    generated = np.array(generated)
    plt.figure()
    plt.title("Mouse Movement: Recorded vs Generated")
    plt.plot(np.cumsum(data[:, 0]), np.cumsum(data[:, 1]), label="Recorded Path", color="blue")
    plt.plot(np.cumsum(generated[:, 0]), np.cumsum(generated[:, 1]), label="Generated Path", color="red")
    plt.legend()
    plt.show()


# =============================
#   Menu
# =============================
def main():
    while True:
        print("\n🎮 MIMIC Control Panel")
        print("1. 🖱️ Collect mouse data")
        print("2. 🧠 Train LSTM model")
        print("3. 🤖 Run cursor spoofer (Live + Graph)")
        print("4. ❌ Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            collect_data()
        elif choice == "2":
            train_lstm("data/mouse_data.csv")
        elif choice == "3":
            spoof_and_plot()
        elif choice == "4":
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
