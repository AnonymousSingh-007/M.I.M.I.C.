#mimic/spoofer.py
#fixed graph
import time
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import pyautogui
from mimic.model import LSTMModel
from rich.console import Console
import re

console = Console()

def load_model_and_scaler(model_path, scaler_path, device="cpu"):
    """
    Load trained LSTM + scaler.
    """
    scaler = joblib.load(scaler_path)
    state = torch.load(model_path, map_location=device)

    num_layers = 0
    for k in state.keys():
        if 'lstm.weight_ih_l' in k:
            nums = re.findall(r'\d+', k)
            if nums:
                num_layers = max(num_layers, int(nums[0]) + 1)

    fc_weight = None
    for k, v in state.items():
        if k.endswith("fc.weight"):
            fc_weight = v
            break
    if fc_weight is None:
        raise RuntimeError("Couldn't find fc.weight in checkpoint.")
    output_size = fc_weight.shape[0]

    model = LSTMModel(input_size=5, hidden_size=128,
                      num_layers=num_layers, output_size=int(output_size))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, scaler

def spoof_and_plot(model_path="models/mimic_lstm.pt",
                   scaler_path="models/mimic_scaler.pkl",
                   csv_path="data/mouse_data.csv",
                   seq_len=70, steps=1000, move_delay=0.001):
    """
    Run spoofer with curvature (using last known curvature as placeholder).
    """
    pyautogui.FAILSAFE = False
    screen_width, screen_height = pyautogui.size()
    left_bound = screen_width * 0.1
    right_bound = screen_width * 0.9
    top_bound = screen_height * 0.1
    bottom_bound = screen_height * 0.9

    device = "cpu"
    model, scaler = load_model_and_scaler(model_path, scaler_path, device=device)
    horizon = int(model.fc.out_features / 2)

    df = pd.read_csv(csv_path)
    features = df[["dx", "dy", "speed", "dt", "curvature"]].values.astype(np.float32)
    features_scaled = scaler.transform(features)
    if len(features_scaled) < seq_len:
        raise ValueError(f"Not enough data ({len(features_scaled)}) for seq_len={seq_len}.")

    seq = torch.tensor(features_scaled[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
    recorded_deltas = features[:, :2]
    generated = []
    last_dt = float(features[-1, 3]) if features.shape[0] > 0 else 1e-3
    last_curvature = float(features[-1, 4]) if features.shape[0] > 0 else 0.0
    current_x, current_y = pyautogui.position()

    console.print("🎮 Spoofing mouse now...")
    try:
        with torch.no_grad():
            for _ in range(steps):
                preds_scaled = model(seq).cpu().numpy()[0]
                preds = np.reshape(preds_scaled, (horizon, 2))

                for pred_dx_s, pred_dy_s in preds:
                    # inverse transform (with placeholder for speed/dt/curvature)
                    row_scaled = np.array([[pred_dx_s, pred_dy_s, 0.0, 0.0, last_curvature]], dtype=np.float32)
                    row_unscaled = scaler.inverse_transform(row_scaled)[0]
                    pred_dx, pred_dy = float(row_unscaled[0]), float(row_unscaled[1])

                    current_x += pred_dx
                    current_y += pred_dy

                    # clamp to bounds
                    current_x = max(left_bound, min(right_bound, current_x))
                    current_y = max(top_bound, min(bottom_bound, current_y))

                    pyautogui.moveTo(current_x, current_y)
                    time.sleep(move_delay)
                    generated.append([pred_dx, pred_dy])

                pred_dx, pred_dy = generated[-1]
                pred_speed = float(np.hypot(pred_dx, pred_dy))
                new_feature_unscaled = np.array([[pred_dx, pred_dy, pred_speed, last_dt, last_curvature]], dtype=np.float32)
                new_feature_scaled = scaler.transform(new_feature_unscaled)[0]
                new_feat_tensor = torch.tensor(new_feature_scaled, dtype=torch.float32, device=device).view(1, 1, -1)
                seq = torch.cat([seq[:, 1:, :], new_feat_tensor], dim=1)

    except pyautogui.FailSafeException:
        console.print("[red]❌ Fail-safe triggered. Spoofing stopped.")
        return

    console.print("✅ Spoofing finished. Showing graph...")

    generated = np.array(generated)

    # --- Convert both recorded and generated into absolute positions with safe margins ---
    start_x, start_y = screen_width // 2, screen_height // 2  # reference start
    rec_positions = [(start_x, start_y)]
    for dx, dy in recorded_deltas:
        new_x = int(max(left_bound, min(right_bound, rec_positions[-1][0] + dx)))
        new_y = int(max(top_bound, min(bottom_bound, rec_positions[-1][1] + dy)))
        rec_positions.append((new_x, new_y))
    rec_positions = np.array(rec_positions)

    gen_positions = [(start_x, start_y)]
    for dx, dy in generated:
        new_x = int(max(left_bound, min(right_bound, gen_positions[-1][0] + dx)))
        new_y = int(max(top_bound, min(bottom_bound, gen_positions[-1][1] + dy)))
        gen_positions.append((new_x, new_y))
    gen_positions = np.array(gen_positions)

    # --- Plot with same margins ---
    plt.figure(figsize=(6, 6))
    plt.title("Mouse Movement: Recorded vs Generated")
    plt.plot(rec_positions[:, 0], rec_positions[:, 1], label="Recorded Path", color="blue")
    plt.plot(gen_positions[:, 0], gen_positions[:, 1], label="Generated Path", color="red", linestyle="--")

    plt.xlim(left_bound, right_bound)
    plt.ylim(top_bound, bottom_bound)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.xlabel("X Position (clamped)")
    plt.ylabel("Y Position (clamped)")
    plt.grid(True)
    plt.show()
