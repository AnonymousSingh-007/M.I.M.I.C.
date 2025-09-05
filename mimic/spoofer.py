#mimic/spoofer.py
import time
import pandas as pd
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import pyautogui
from mimic.model import LSTMModel

# =============================
#   Load Model + Scaler
# =============================
def load_model_and_scaler(model_path, scaler_path, device="cpu"):
    """
    Load trained LSTM + scaler.
    Infers output_size (horizon*2) from saved weights.
    """
    scaler = joblib.load(scaler_path)
    state = torch.load(model_path, map_location=device)

    # infer output size from fc layer
    fc_weight = None
    for k, v in state.items():
        if k.endswith("fc.weight"):
            fc_weight = v
            break
    if fc_weight is None:
        raise RuntimeError("Couldn't find fc.weight in checkpoint.")
    output_size = fc_weight.shape[0]

    model = LSTMModel(input_size=4, hidden_size=128, num_layers=2, output_size=int(output_size))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, scaler


# =============================
#   Spoofing + Graph
# =============================
def spoof_and_plot(model_path="models/mimic_lstm.pt",
                   scaler_path="models/mimic_scaler.pkl",
                   csv_path="data/mouse_data.csv",
                   seq_len=70, # Adjusted to match your config
                   steps=1000, # Increased for a longer path
                   move_delay=0.001):
    """
    Run spoofer with boundaries to prevent fail-safe.
      - Uses pre-computed features from CSV
      - Seeds last seq_len frames
      - Generates multiple predicted steps per loop
    """
    # Set a custom failsafe region in the center of the screen
    # These are percentages of the screen size
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
    features = df[["dx", "dy", "speed", "dt"]].values.astype(np.float32)
    
    # scale
    features_scaled = scaler.transform(features)
    if len(features_scaled) < seq_len:
        raise ValueError(f"Not enough data ({len(features_scaled)}) for seq_len={seq_len}. Please collect more mouse data.")

    seq = torch.tensor(features_scaled[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)

    recorded_deltas = features[:, :2]  # for plotting
    generated = []
    last_dt = float(features[-1, 3]) if features.shape[0] > 0 else 1e-3

    current_x, current_y = pyautogui.position()

    print("🎮 Spoofing mouse now...")
    try:
        with torch.no_grad():
            for _ in range(steps):
                preds_scaled = model(seq).cpu().numpy()[0]
                
                # Predict horizon steps, one by one
                current_preds = np.reshape(preds_scaled, (horizon, 2))
                
                for pred_dx_s, pred_dy_s in current_preds:
                    # inverse transform (fill with zeros for speed/dt)
                    row_scaled = np.array([[pred_dx_s, pred_dy_s, 0.0, 0.0]], dtype=np.float32)
                    row_unscaled = scaler.inverse_transform(row_scaled)[0]
                    pred_dx, pred_dy = float(row_unscaled[0]), float(row_unscaled[1])

                    # calculate next position based on prediction
                    next_x = current_x + pred_dx
                    next_y = current_y + pred_dy

                    # Enforce the boundaries. Clamp to the boundary and reverse direction if needed.
                    if next_x < left_bound:
                        next_x = left_bound
                        pred_dx = 0 # stop moving horizontally
                    elif next_x > right_bound:
                        next_x = right_bound
                        pred_dx = 0
                    
                    if next_y < top_bound:
                        next_y = top_bound
                        pred_dy = 0 # stop moving vertically
                    elif next_y > bottom_bound:
                        next_y = bottom_bound
                        pred_dy = 0

                    current_x = next_x
                    current_y = next_y

                    pyautogui.moveTo(current_x, current_y)
                    time.sleep(move_delay)

                    generated.append([pred_dx, pred_dy])

                # build next feature for sequence
                pred_dx, pred_dy = generated[-1]
                pred_speed = float(np.hypot(pred_dx, pred_dy))
                new_feature_unscaled = np.array([[pred_dx, pred_dy, pred_speed, last_dt]], dtype=np.float32)
                new_feature_scaled = scaler.transform(new_feature_unscaled)[0]

                new_feat_tensor = torch.tensor(new_feature_scaled, dtype=torch.float32, device=device).view(1, 1, -1)
                seq = torch.cat([seq[:, 1:, :], new_feat_tensor], dim=1)

    except pyautogui.FailSafeException:
        print("[red]❌ PyAutoGUI fail-safe triggered. Spoofing has been stopped.")
        if not generated:
            print("[yellow]No movements were generated before the fail-safe was triggered.")
            return

    print("✅ Spoofing finished. Showing graph...")

    # === Plot ===
    generated = np.array(generated)
    rec_cum_x, rec_cum_y = np.cumsum(recorded_deltas[:, 0]), np.cumsum(recorded_deltas[:, 1])
    gen_cum_x, gen_cum_y = np.cumsum(generated[:, 0]), np.cumsum(generated[:, 1])

    plt.figure()
    plt.title("Mouse Movement: Recorded vs Generated")
    plt.plot(rec_cum_x, rec_cum_y, label="Recorded Path")
    plt.plot(gen_cum_x, gen_cum_y, label="Generated Path")
    plt.legend()
    plt.xlabel("Cumulative dx")
    plt.ylabel("Cumulative dy")
    plt.axis('equal') # better scaling
    plt.grid(True)
    plt.show()

