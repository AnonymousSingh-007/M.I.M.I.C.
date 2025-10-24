#mimic/spoofer.py

# import time
# import pandas as pd
# import numpy as np
# import torch
# import joblib
# import matplotlib.pyplot as plt
# import pyautogui
# from mimic.model import LSTMModel
# from rich.console import Console
# import re

# console = Console()

# def load_model_and_scaler(model_path, scaler_path, device="cpu"):
#     """
#     Load trained LSTM + scaler.
#     """
#     scaler = joblib.load(scaler_path)
#     state = torch.load(model_path, map_location=device)

#     num_layers = 0
#     for k in state.keys():
#         if 'lstm.weight_ih_l' in k:
#             nums = re.findall(r'\d+', k)
#             if nums:
#                 num_layers = max(num_layers, int(nums[0]) + 1)

#     fc_weight = None
#     for k, v in state.items():
#         if k.endswith("fc.weight"):
#             fc_weight = v
#             break
#     if fc_weight is None:
#         raise RuntimeError("Couldn't find fc.weight in checkpoint.")
#     output_size = fc_weight.shape[0]

#     model = LSTMModel(input_size=5, hidden_size=128,
#                       num_layers=num_layers, output_size=int(output_size))
#     model.load_state_dict(state)
#     model.to(device)
#     model.eval()
#     return model, scaler

# def spoof_and_plot(model_path="models/mimic_lstm.pt",
#                    scaler_path="models/mimic_scaler.pkl",
#                    csv_path="data/mouse_data.csv",
#                    seq_len=70, steps=1000, move_delay=0.001):
#     """
#     Run spoofer with curvature (using last known curvature as placeholder).
#     """
#     pyautogui.FAILSAFE = False
#     screen_width, screen_height = pyautogui.size()
#     left_bound = screen_width * 0.1
#     right_bound = screen_width * 0.9
#     top_bound = screen_height * 0.1
#     bottom_bound = screen_height * 0.9

#     device = "cpu"
#     model, scaler = load_model_and_scaler(model_path, scaler_path, device=device)
#     pred_horizon = int(model.fc.out_features / 2)

#     df = pd.read_csv(csv_path)
#     features = df[["dx", "dy", "speed", "dt", "curvature"]].values.astype(np.float32)
#     features_scaled = scaler.transform(features)
#     if len(features_scaled) < seq_len:
#         raise ValueError(f"Not enough data ({len(features_scaled)}) for seq_len={seq_len}.")

#     seq = torch.tensor(features_scaled[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
#     recorded_deltas = features[:, :2]
#     generated = []
#     last_dt = float(features[-1, 3]) if features.shape[0] > 0 else 1e-3
#     last_curvature = float(features[-1, 4]) if features.shape[0] > 0 else 0.0
#     current_x, current_y = pyautogui.position()

#     console.print("🎮 Spoofing mouse now...")
#     try:
#         with torch.no_grad():
#             for _ in range(steps):
#                 preds_scaled = model(seq).cpu().numpy()[0]
#                 preds = np.reshape(preds_scaled, (pred_horizon, 2))

#                 for pred_dx_s, pred_dy_s in preds:
#                     # inverse transform (with placeholder for speed/dt/curvature)
#                     row_scaled = np.array([[pred_dx_s, pred_dy_s, 0.0, 0.0, last_curvature]], dtype=np.float32)
#                     row_unscaled = scaler.inverse_transform(row_scaled)[0]
#                     pred_dx, pred_dy = float(row_unscaled[0]), float(row_unscaled[1])

#                     current_x += pred_dx
#                     current_y += pred_dy

#                     # clamp to bounds
#                     current_x = max(left_bound, min(right_bound, current_x))
#                     current_y = max(top_bound, min(bottom_bound, current_y))

#                     pyautogui.moveTo(current_x, current_y)
#                     time.sleep(move_delay)
#                     generated.append([pred_dx, pred_dy])

#                 pred_dx, pred_dy = generated[-1]
#                 pred_speed = float(np.hypot(pred_dx, pred_dy))
#                 new_feature_unscaled = np.array([[pred_dx, pred_dy, pred_speed, last_dt, last_curvature]], dtype=np.float32)
#                 new_feature_scaled = scaler.transform(new_feature_unscaled)[0]
#                 new_feat_tensor = torch.tensor(new_feature_scaled, dtype=torch.float32, device=device).view(1, 1, -1)
#                 seq = torch.cat([seq[:, 1:, :], new_feat_tensor], dim=1)

#     except pyautogui.FailSafeException:
#         console.print("[red]❌ Fail-safe triggered. Spoofing stopped.")
#         return

#     console.print("✅ Spoofing finished. Showing graph...")

#     generated = np.array(generated)

#     # --- Convert both recorded and generated into absolute positions with safe margins ---
#     start_x, start_y = screen_width // 2, screen_height // 2  # reference start
#     rec_positions = [(start_x, start_y)]
#     for dx, dy in recorded_deltas:
#         new_x = int(max(left_bound, min(right_bound, rec_positions[-1][0] + dx)))
#         new_y = int(max(top_bound, min(bottom_bound, rec_positions[-1][1] + dy)))
#         rec_positions.append((new_x, new_y))
#     rec_positions = np.array(rec_positions)

#     gen_positions = [(start_x, start_y)]
#     for dx, dy in generated:
#         new_x = int(max(left_bound, min(right_bound, gen_positions[-1][0] + dx)))
#         new_y = int(max(top_bound, min(bottom_bound, gen_positions[-1][1] + dy)))
#         gen_positions.append((new_x, new_y))
#     gen_positions = np.array(gen_positions)

#     # --- Plot with same margins ---
#     plt.figure(figsize=(6, 6))
#     plt.title("Mouse Movement: Recorded vs Generated")
#     plt.plot(rec_positions[:, 0], rec_positions[:, 1], label="Recorded Path", color="blue")
#     plt.plot(gen_positions[:, 0], gen_positions[:, 1], label="Generated Path", color="red", linestyle="--")

#     plt.xlim(left_bound, right_bound)
#     plt.ylim(top_bound, bottom_bound)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.legend()
#     plt.xlabel("X Position (clamped)")
#     plt.ylabel("Y Position (clamped)")
#     plt.grid(True)
#     plt.show()

# mimic/spoofer.py
# mimic/spoofer.py
# import time
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# import pyautogui
# from rich.console import Console
# from mimic.model import Seq2SeqModel
# from mimic.visuals import display_status, display_success, display_error

# console = Console()


# def spoof_and_collect(
#     model_path: str,
#     scaler_path: str,
#     csv_path: str,
#     seq_len: int = 100,
#     pred_horizon: int = 20,  # ignored here; we generate for total_steps
#     hidden_size: int = 128,
#     num_layers: int = 2,
#     device: str = "cpu",
#     **kwargs,  # <--- absorb extras like model_type
# ):
#     """
#     Load model+scaler, autoregressively generate as many steps as available
#     (len(df) - seq_len), move the mouse using predicted dx/dy and predicted dt,
#     and return recorded_deltas and generated_deltas for plotting.
#     """
#     try:
#         display_status(f"Loading model from {model_path}")
#         scaler = joblib.load(scaler_path)

#         df = pd.read_csv(csv_path)
#         # Expect df columns: ["dx","dy","speed","dt","curvature","x_norm","y_norm"]
#         feature_names = ["dx", "dy", "speed", "dt", "curvature", "x_norm", "y_norm"]
#         if list(df.columns[: len(feature_names)]) != feature_names:
#             # If your df has exactly those columns but in different order, adjust here.
#             console.print("[yellow]Warning: CSV columns don't exactly match expected feature names.[/yellow]")

#         input_size = len(feature_names)

#         model = Seq2SeqModel(
#             input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=input_size
#         ).to(device)
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.eval()

#         display_success("Model loaded")

#         # validate we have enough rows for a seed
#         if len(df) <= seq_len:
#             display_error("Not enough data to seed the model (df rows <= seq_len).")
#             return None, None

#         # Number of steps to generate to match recorded length
#         total_steps = len(df) - seq_len

#         # Seed is the last seq_len rows (use same feature order as training)
#         seed_df = df[feature_names].tail(seq_len)
#         seed_seq = seed_df.values.astype(np.float32)  # shape (seq_len, 7)

#         # Generate total_steps predictions at once (model.generate returns unscaled features)
#         display_status(f"Generating {total_steps} steps...")
#         generated = model.generate(seed_seq, scaler, pred_horizon=total_steps, device=device)
#         # generated should be shape (total_steps, input_size)
#         if generated.ndim != 2 or generated.shape[1] < 2:
#             display_error("Generated output shape unexpected.")
#             return None, None

#         # Prepare recorded deltas to compare: the ground-truth rows that come after the seed.
#         recorded_deltas = df[feature_names].values[seq_len:, 0:2]  # dx, dy from real data
#         generated_deltas = generated[:, 0:2]  # dx, dy from model

#         # Now move the mouse according to generated deltas
#         pyautogui.FAILSAFE = True
#         screen_w, screen_h = pyautogui.size()
#         left_bound = screen_w * 0.01
#         right_bound = screen_w * 0.99
#         top_bound = screen_h * 0.01
#         bottom_bound = screen_h * 0.99

#         current_x, current_y = pyautogui.position()
#         display_status("Starting live cursor spoof (move your mouse to a corner to abort)...")

#         try:
#             for idx, row in enumerate(generated):
#                 # row layout: [dx, dy, speed, dt, curvature, x_norm, y_norm]
#                 dx = float(row[0])
#                 dy = float(row[1])
#                 # speed = float(row[2])  # currently unused for movement; could be used for smoothing
#                 dt = float(row[3]) if not np.isnan(row[3]) else 0.001

#                 current_x += dx
#                 current_y += dy

#                 # clamp inside bounds
#                 current_x = max(left_bound, min(right_bound, current_x))
#                 current_y = max(top_bound, min(bottom_bound, current_y))

#                 # move the cursor (use int screen coords)
#                 pyautogui.moveTo(int(round(current_x)), int(round(current_y)))

#                 # sleep according to predicted dt, enforce safe minimum
#                 time.sleep(max(0.001, dt))

#             display_success(f"Spoofing finished: {len(generated)} steps applied.")
#         except pyautogui.FailSafeException:
#             display_error("Fail-safe triggered — spoofing stopped by moving mouse to corner.")
#             # still return what we have so far
#             # Note: If fail-safe triggers, the OS might have moved the cursor; plot may be partial.
#             return recorded_deltas, generated_deltas

#         return recorded_deltas, generated_deltas

#     except Exception as e:
#         display_error(f"Unexpected error: {e}")
#         return None, None



#gemini approach

# mimic/spoofer.py

# import time
# import joblib
# import numpy as np
# import pandas as pd
# import torch
# import pyautogui
# from rich.console import Console
# from mimic.model import Seq2SeqModel
# from mimic.visuals import display_status, display_success, display_error

# console = Console()

# def spoof_and_collect(
#     model_path: str,
#     scaler_path: str,
#     csv_path: str,
#     seq_len: int = 100,
#     pred_horizon: int = 15,
#     hidden_size: int = 128,
#     num_layers: int = 2,
#     **kwargs,
# ):
#     """
#     Load model+scaler, autoregressively generate movements in chunks,
#     move the mouse using predicted dx/dy + dt,
#     and return recorded_deltas and generated_deltas for plotting.
#     """
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     try:
#         display_status(f"Loading model from {model_path}")
#         scaler = joblib.load(scaler_path)

#         df = pd.read_csv(csv_path)
#         feature_names = list(df.columns)
#         input_size = len(feature_names)

#         model = Seq2SeqModel(
#             input_size=input_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             output_size=input_size,
#         ).to(device)
#         model.load_state_dict(torch.load(model_path, map_location=device))
#         model.eval()

#         display_success("Model loaded")

#         if len(df) <= seq_len:
#             display_error("Not enough data to seed the model.")
#             return None, None

#         # === Generate deltas ===
#         generated_deltas = []
#         current_seed = df.tail(seq_len).values.astype(np.float32)

#         total_steps = len(df) - seq_len
#         num_chunks = total_steps // pred_horizon + (1 if total_steps % pred_horizon != 0 else 0)

#         for _ in range(num_chunks):
#             gen_chunk = model.generate(current_seed, scaler, horizon=pred_horizon)
#             generated_deltas.extend(gen_chunk.tolist())
#             generated_scaled = scaler.transform(gen_chunk)
#             current_seed = np.vstack([current_seed[pred_horizon:], generated_scaled])

#         generated_deltas = np.array(generated_deltas)

#         # === Recorded data for plotting ===
#         recorded_deltas_source = df.values[seq_len:, :len(feature_names)]
#         recorded_denormalized = scaler.inverse_transform(recorded_deltas_source)
#         rec_deltas_pixels = recorded_denormalized[:, 0:2]

#         screen_w, screen_h = pyautogui.size()
#         rec_deltas_pixels[:, 0] *= screen_w
#         rec_deltas_pixels[:, 1] *= screen_h

#         # === Cursor movement ===
#         display_status("Starting live cursor spoof...")

#         pyautogui.FAILSAFE = False
#         left_bound, right_bound = screen_w * 0.01, screen_w * 0.99
#         top_bound, bottom_bound = screen_h * 0.01, screen_h * 0.99

#         current_x, current_y = pyautogui.position()

#         try:
#             for row in generated_deltas:
#                 dx = int(round(float(row[0] * screen_w)))
#                 dy = int(round(float(row[1] * screen_h)))
#                 dt = float(row[3]) if not np.isnan(row[3]) else 0.001

#                 current_x = min(max(current_x + dx, left_bound), right_bound)
#                 current_y = min(max(current_y + dy, top_bound), bottom_bound)

#                 pyautogui.moveTo(int(current_x), int(current_y))
#                 time.sleep(max(0.001, dt))

#             display_success(f"Spoofing finished: {total_steps} steps applied.")

#         except Exception as e:
#             display_error(f"Error during spoofing: {e}")
#         finally:
#             pyautogui.FAILSAFE = True

#         return rec_deltas_pixels, generated_deltas[:, 0:2]

#     except Exception as e:
#         display_error(f"Unexpected error: {e}")
#         return None, None












#mimic/spoofer.py - E




# mimic/spoofer.py 

# mimic/spoofer.py
# mimic/spoofer.py
# mimic/spoofer.py
import torch
import pandas as pd
import numpy as np
import pyautogui
from mimic.lstm_trainer import load_model_and_scaler
import time
import matplotlib.pyplot as plt
from mimic import visuals

def spoof_and_plot(model_path, scaler_path, csv_path, seq_len=120, steps=800, move_delay=0.001):
    """
    Run cursor spoofer using trained LSTM and plot generated vs recorded paths.
    Includes safe boundaries and gamified progress messages.
    """
    pyautogui.FAILSAFE = False
    device = "cpu"

    # Load model + scaler
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    horizon = int(model.fc.out_features / 2)

    # Screen boundaries (10% margins)
    screen_width, screen_height = pyautogui.size()
    left_bound = screen_width * 0.1
    right_bound = screen_width * 0.9
    top_bound = screen_height * 0.1
    bottom_bound = screen_height * 0.9

    # Load recorded features
    df = pd.read_csv(csv_path)
    if {"dx", "dy", "speed", "dt"}.issubset(df.columns):
        features = df[["dx", "dy", "speed", "dt"]].values.astype(np.float32)
    else:
        coords = df[["x", "y"]].values
        deltas = np.diff(coords, axis=0)
        dt = np.diff(df["time"].values)
        dt = np.where(dt <= 0, 1e-3, dt)
        speed = np.linalg.norm(deltas, axis=1)
        features = np.column_stack([deltas, speed, dt])

    if len(features) < seq_len:
        raise ValueError(f"Not enough data ({len(features)}) for seq_len={seq_len}.")

    # Seed initial sequence
    seq = torch.tensor(features[-seq_len:], dtype=torch.float32, device=device).unsqueeze(0)
    last_dt = float(features[-1, 3])
    current_x, current_y = pyautogui.position()
    generated = []

    visuals.display_status("🎮 Spoofing mouse now...")

    with torch.no_grad():
        for step in range(steps):
            # Predict horizon*2 deltas
            pred_scaled = model(seq).cpu().numpy()[0]
            pred_dx_scaled, pred_dy_scaled = pred_scaled[:2]

            # Build 4-feature row for inverse_transform
            speed = np.hypot(pred_dx_scaled, pred_dy_scaled)
            row_scaled = np.array([[pred_dx_scaled, pred_dy_scaled, speed, last_dt]], dtype=np.float32)

            # Inverse transform to original scale
            pred_dx, pred_dy = scaler.inverse_transform(row_scaled)[0][:2]

            # Update cursor position with boundaries
            next_x = np.clip(current_x + pred_dx, left_bound, right_bound)
            next_y = np.clip(current_y + pred_dy, top_bound, bottom_bound)

            # Stop movement if boundary reached
            pred_dx = next_x - current_x
            pred_dy = next_y - current_y

            current_x, current_y = next_x, next_y
            pyautogui.moveTo(current_x, current_y)
            generated.append([pred_dx, pred_dy])

            # Prepare next sequence
            new_feat_scaled = scaler.transform(
                np.array([[pred_dx, pred_dy, np.hypot(pred_dx, pred_dy), last_dt]], dtype=np.float32)
            )[0]
            seq = torch.cat([seq[:, 1:, :], torch.tensor(new_feat_scaled, dtype=torch.float32, device=device).view(1, 1, -1)], dim=1)

            # Optional gamified progress display every 50 steps
            if step % 50 == 0 and step > 0:
                visuals.display_status(f"→ {step}/{steps} steps completed...")

    visuals.display_success("✅ Spoofing finished. Displaying graph...")

    # Plot recorded vs generated path
    generated = np.array(generated)
    rec_cum_x, rec_cum_y = np.cumsum(features[:, 0]), np.cumsum(features[:, 1])
    gen_cum_x, gen_cum_y = np.cumsum(generated[:, 0]), np.cumsum(generated[:, 1])

    plt.figure(figsize=(8, 6))
    plt.title("Mouse Movement: Recorded vs Generated")
    plt.plot(rec_cum_x, rec_cum_y, label="Recorded Path", color="blue")
    plt.plot(gen_cum_x, gen_cum_y, label="Generated Path", color="red")
    plt.legend()
    plt.xlabel("Cumulative dx")
    plt.ylabel("Cumulative dy")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
