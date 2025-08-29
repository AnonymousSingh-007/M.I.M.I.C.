
# # import time
# # import csv
# # import os
# # import random
# # import pyautogui
# # from pathlib import Path
# # from pynput import mouse
# # from rich.console import Console
# # import torch
# # import math

# # console = Console()

# # DATA_DIR = "data"
# # DATA_FILE = os.path.join(DATA_DIR, "mouse_movements.csv")


# # # 🎯 Record mouse movements
# # def record_mouse_movements(duration=15):
# #     Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
# #     with open(DATA_FILE, mode="w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow(["time", "dx", "dy", "speed", "pause", "x", "y"])

# #         start_time = time.time()
# #         prev_time = start_time
# #         prev_x, prev_y = pyautogui.position()

# #         def on_move(x, y):
# #             nonlocal prev_time, prev_x, prev_y
# #             now = time.time()
# #             if now - start_time > duration:
# #                 return False

# #             dx = x - prev_x
# #             dy = y - prev_y
# #             dt = now - prev_time if now - prev_time > 0 else 1e-6
# #             speed = math.sqrt(dx**2 + dy**2) / dt
# #             pause = 1 if dx == 0 and dy == 0 else 0

# #             rel_time = now - start_time
# #             writer.writerow([rel_time, dx, dy, speed, pause, x, y])

# #             prev_x, prev_y = x, y
# #             prev_time = now
# #             return True

# #         console.print(f"🎙️ Recording mouse for {duration} seconds...")
# #         with mouse.Listener(on_move=on_move) as listener:
# #             listener.join()

# #     console.print(f"✅ Data saved to: {DATA_FILE}")


# # # 🤖 Simulate cursor movement (now more human-like)
# # def simulate_movement(model, duration=5):
# #     console.print(f"🌀 Spoofing for {duration} seconds...")
# #     console.print("🖱️ Releasing control to M.I.M.I.C...")

# #     pyautogui.FAILSAFE = False
# #     start_time = time.time()
# #     prev_time = start_time
# #     prev_x, prev_y = pyautogui.position()

# #     screen_width, screen_height = pyautogui.size()
# #     margin = 5

# #     while time.time() - start_time < duration:
# #         now = time.time()
# #         rel_time = now - start_time

# #         dx = prev_x - prev_x
# #         dy = prev_y - prev_y
# #         dt = now - prev_time if now - prev_time > 0 else 1e-6
# #         speed = math.sqrt(dx**2 + dy**2) / dt
# #         pause = 1 if dx == 0 and dy == 0 else 0

# #         # Build and normalize input vector
# #         input_vector = torch.tensor([[rel_time, dx, dy, speed, pause]], dtype=torch.float32)
# #         input_vector = (input_vector - torch.tensor(model.mean)) / torch.tensor(model.std)

# #         # Predict target position
# #         with torch.no_grad():
# #             pred_x, pred_y = model(input_vector).squeeze().tolist()

# #         # Clamp within screen
# #         pred_x = max(margin, min(screen_width - margin, pred_x))
# #         pred_y = max(margin, min(screen_height - margin, pred_y))

# #         # --- Human-like movement adjustments ---
# #         jitter_x = random.uniform(-1.5, 1.5)
# #         jitter_y = random.uniform(-1.5, 1.5)
# #         pred_x += jitter_x
# #         pred_y += jitter_y

# #         # Curved path effect (blend with previous pos)
# #         curve_factor = random.uniform(0.05, 0.2)
# #         pred_x = prev_x + (pred_x - prev_x) * (1 - curve_factor)
# #         pred_y = prev_y + (pred_y - prev_y) * (1 - curve_factor)

# #         # Variable movement delay
# #         move_duration = random.uniform(0.015, 0.08)
# #         pyautogui.moveTo(pred_x, pred_y, duration=move_duration)

# #         # Occasional micro-pause
# #         if random.random() < 0.1:
# #             time.sleep(random.uniform(0.02, 0.06))

# #         prev_x, prev_y = pred_x, pred_y
# #         prev_time = now

# #with lstm
# import csv
# import time
# from typing import Optional

# import pyautogui
# from pynput import mouse, keyboard
# import torch
# import numpy as np
# from mimic.model import load_model_and_scaler


# def collect_data(path: str, duration: Optional[int] = None) -> None:
#     """
#     Collect mouse movement data into CSV file.
#     Press ESC to stop early.
#     """
#     print(f"📡 Collecting mouse movement data -> {path}")

#     stop_flag = {"stop": False}
#     start_time = time.time()

#     def on_press(key) -> None:  # always returns None
#         if key == keyboard.Key.esc:
#             stop_flag["stop"] = True
#             # Stop keyboard listener
#             return None

#     with open(path, mode="w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["time", "x", "y"])

#         def on_move(x, y):
#             if stop_flag["stop"]:
#                 return False  # ✅ mouse.Listener *does* allow False to stop
#             writer.writerow([time.time() - start_time, x, y])
#             return None

#         with mouse.Listener(on_move=on_move) as listener, \
#              keyboard.Listener(on_press=on_press) as k_listener:
#             while listener.running and not stop_flag["stop"]:
#                 if duration and (time.time() - start_time) > duration:
#                     stop_flag["stop"] = True
#                     break
#                 time.sleep(0.01)


# def predict_sequence(model, scaler, seed_sequence, pred_len: int = 50):
#     """Predict next sequence of (x, y) coords given a seed sequence"""
#     model.eval()
#     predictions = []

#     seq = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0)
#     for _ in range(pred_len):
#         with torch.no_grad():
#             next_point = model(seq).numpy()
#         predictions.append(next_point)
#         seq = torch.cat([seq[:, 1:, :], torch.tensor(next_point).unsqueeze(0)], dim=1)

#     predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 2))
#     return predictions.tolist()


# def live_path_tracing(model_path="models/mimic_lstm.pt", seq_len=30):
#     """
#     Run cursor spoofer with trained LSTM + live tracing
#     """
#     model, scaler = load_model_and_scaler(model_path)

#     # grab current position as seed
#     x, y = pyautogui.position()
#     seed = [[x, y]] * seq_len
#     seed_scaled = scaler.transform(seed)

#     path = predict_sequence(model, scaler, seed_scaled, pred_len=100)

#     print("🚀 Running live path tracing...")
#     for (px, py) in path:
#         pyautogui.moveTo(int(px), int(py), duration=0.02)

#lstm + anime
# mimic/spoofer.py
import csv
import time
import os
import random
from typing import Optional, List, Tuple

import pyautogui
from pynput import mouse, keyboard
import torch
import numpy as np
from rich.progress import Progress

# Try to import the voice helper from places you might have it.
try:
    # if you placed voice.py at assets/sounds/voice.py
    from assets.sounds.voice import speak_anime_line  # type: ignore
except Exception:
    try:
        # or if you put it under mimic/voice.py
        from mimic.voice import speak_anime_line  # type: ignore
    except Exception:
        # fallback no-op if not available
        def speak_anime_line() -> None:
            return None


# Model loader (expects models/scaler.pkl saved by training)
from mimic.model import load_model_and_scaler


# -------------------------
# Data collection
# -------------------------
def collect_data(path: str, duration: Optional[int] = None) -> None:
    """
    Collect mouse movement data into CSV file.
    - path: where to save CSV
    - duration: seconds to record (if None -> until ESC pressed)
    Press ESC to stop early.
    CSV header: time,x,y
    Shows a rich progress bar if duration is provided.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"📡 Collecting mouse movement data -> {path}")

    stop_flag = {"stop": False}
    start_time = time.time()

    def on_press(key) -> None:
        # keyboard.Listener expects a callback that returns None
        # we set a flag instead to stop the loop/listener
        if key == keyboard.Key.esc:
            stop_flag["stop"] = True
        return None

    with open(path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "x", "y"])

        def on_move(x, y):
            if stop_flag["stop"]:
                return False  # this stops the mouse listener
            writer.writerow([time.time() - start_time, x, y])
            return None

        with mouse.Listener(on_move=on_move) as m_listener, \
             keyboard.Listener(on_press=on_press) as k_listener:

            # If duration is provided, show progress; otherwise run until ESC
            if duration and duration > 0:
                with Progress() as progress:
                    task = progress.add_task("[cyan]Recording mouse data...", total=duration)
                    while m_listener.running and not stop_flag["stop"]:
                        elapsed = time.time() - start_time
                        progress.update(task, completed=min(elapsed, duration))
                        if elapsed >= duration:
                            stop_flag["stop"] = True
                            break
                        time.sleep(0.05)
            else:
                # Indefinite recording until ESC
                print("Recording... press ESC to stop.")
                while m_listener.running and not stop_flag["stop"]:
                    time.sleep(0.05)

    print(f"✅ Data saved to: {path}")


# -------------------------
# Helpers for reading CSV seed / duration
# -------------------------
def get_recorded_duration(path: str) -> Optional[float]:
    """
    Read the CSV and return the recorded duration in seconds.
    Supports CSVs with header 'time' (relative) or 'timestamp' (epoch).
    Returns None if it cannot determine duration.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            if not headers:
                return None

            # prefer 'time' (relative)
            if "time" in headers:
                t_idx = headers.index("time")
                last = None
                for row in reader:
                    if not row:
                        continue
                    try:
                        last = float(row[t_idx])
                    except Exception:
                        pass
                return last if last is not None else None

            # if only timestamps are present, compute last - first
            if "timestamp" in headers:
                t_idx = headers.index("timestamp")
                first = None
                last = None
                for row in reader:
                    if not row:
                        continue
                    try:
                        v = float(row[t_idx])
                    except Exception:
                        continue
                    if first is None:
                        first = v
                    last = v
                if first is not None and last is not None:
                    return float(last - first)
                return None

            # no recognized time column
            return None
    except Exception:
        return None


def _last_seed_xy(path: str, seq_len: int) -> List[List[float]]:
    """
    Return last seq_len [x,y] pairs from CSV (original scale).
    If file too short, pads by repeating the first available row.
    Expects CSV to contain columns named 'x' and 'y' (or 'screen_x','screen_y').
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    xs: List[List[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        if not headers:
            raise ValueError("Empty CSV")

        # choose x,y indices
        if "x" in headers and "y" in headers:
            xi = headers.index("x")
            yi = headers.index("y")
        elif "screen_x" in headers and "screen_y" in headers:
            xi = headers.index("screen_x")
            yi = headers.index("screen_y")
        else:
            # try common older formats: time,timestamp,x,y
            if len(headers) >= 3 and headers[-2].lower() == "x" and headers[-1].lower() == "y":
                xi = len(headers) - 2
                yi = len(headers) - 1
            else:
                raise ValueError("CSV must contain 'x' and 'y' columns")

        for row in reader:
            if not row:
                continue
            try:
                x = float(row[xi])
                y = float(row[yi])
            except Exception:
                continue
            xs.append([x, y])

    if len(xs) == 0:
        raise ValueError("No coordinate rows found in CSV")

    if len(xs) >= seq_len:
        return xs[-seq_len:]
    else:
        # pad by repeating first element
        pad = [xs[0] for _ in range(seq_len - len(xs))]
        return pad + xs


# -------------------------
# Prediction + fallback
# -------------------------
def predict_sequence(model, scaler, seed_sequence: List[List[float]], pred_len: int = 50) -> List[List[float]]:
    """
    Predict next pred_len (x,y) points.
    - seed_sequence must be scaled (same preprocessing used at training).
      (If you pass unscaled seed, call scaler.transform before.)
    Returns list of points in ORIGINAL scale (inverse transformed).
    """
    model.eval()
    preds = []

    # seq shape: (1, seq_len, 2)
    seq = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0)

    for _ in range(pred_len):
        with torch.no_grad():
            out = model(seq)  # expected shape (batch=1, 2)
            out_np = out.detach().cpu().numpy()[0]  # shape (2,)
        preds.append(out_np.copy())

        # append predicted point to seq and drop first
        next_t = torch.tensor(out_np, dtype=torch.float32).view(1, 1, 2)  # (1,1,2)
        seq = torch.cat([seq[:, 1:, :], next_t], dim=1)

    preds_arr = np.array(preds).reshape(-1, 2)
    # inverse transform back to original coordinate scale
    try:
        orig = scaler.inverse_transform(preds_arr)
    except Exception:
        # if scaler breaks for any reason, return raw preds in case
        orig = preds_arr
    return orig.tolist()


def _fallback_procedural_path(seed_xy: List[List[float]], steps: int, jitter_px: float = 1.0) -> List[Tuple[float, float]]:
    """
    Procedural human-ish path fallback if model fails.
    seed_xy: recent points in ORIGINAL scale.
    """
    pts = [tuple(map(float, p)) for p in seed_xy]
    if not pts:
        cx, cy = pyautogui.position()
        pts = [(float(cx), float(cy))]

    if len(pts) == 1:
        base_x, base_y = pts[-1]
        out = []
        x, y = base_x, base_y
        for _ in range(steps):
            x += random.uniform(-3, 3)
            y += random.uniform(-3, 3)
            out.append((x + random.uniform(-jitter_px, jitter_px), y + random.uniform(-jitter_px, jitter_px)))
        return out

    # Use last delta and decay it
    x0, y0 = pts[-2]
    x1, y1 = pts[-1]
    dx = (x1 - x0) * 0.9
    dy = (y1 - y0) * 0.9
    x, y = x1, y1
    out = []
    for _ in range(steps):
        dx *= 0.98
        dy *= 0.98
        x += dx + random.uniform(-jitter_px, jitter_px)
        y += dy + random.uniform(-jitter_px, jitter_px)
        out.append((x, y))
    return out


def _clamp_to_screen(x: float, y: float, margin: int = 5) -> Tuple[int, int]:
    w, h = pyautogui.size()
    xi = max(margin, min(int(round(x)), w - margin - 1))
    yi = max(margin, min(int(round(y)), h - margin - 1))
    return xi, yi


# -------------------------
# Live path tracing
# -------------------------
def live_path_tracing(
    model_path: str = "models/mimic_lstm.pt",
    csv_path: Optional[str] = None,
    seq_len: int = 30,
    duration_sec: Optional[float] = None,
    speed_hz: float = 60.0,
    jitter_px: float = 0.6,
    margin_px: int = 5,
) -> None:
    """
    Run live path tracing:
      - model_path: path to .pt
      - csv_path: path to CSV to seed predictions (if provided/valid)
      - seq_len: number of frames used as seed
      - duration_sec: how many seconds to spoof. If None and csv_path provided, main should compute and pass recorded duration.
      - speed_hz: how many updates per second (affects number of prediction steps)
      - jitter_px: micro-jitter applied to predicted points
    """
    # load model + scaler
    model, scaler = load_model_and_scaler(model_path)

    # choose seed: prefer CSV seed if provided, else use current mouse position repeated
    seed_original: List[List[float]]
    if csv_path and os.path.exists(csv_path):
        try:
            seed_original = _last_seed_xy(csv_path, seq_len)
        except Exception:
            # fallback to current pos
            cx, cy = pyautogui.position()
            seed_original = [[float(cx), float(cy)] for _ in range(seq_len)]
    else:
        cx, cy = pyautogui.position()
        seed_original = [[float(cx), float(cy)] for _ in range(seq_len)]

    # if duration not provided, attempt to read from csv_path
    if duration_sec is None and csv_path:
        duration_read = get_recorded_duration(csv_path)
        if duration_read:
            duration_sec = float(duration_read)

    if duration_sec is None or duration_sec <= 0:
        # safety default: 10 seconds
        duration_sec = 10.0

    # speak an anime line before starting (if available)
    try:
        speak_anime_line()
    except Exception:
        pass
    time.sleep(0.6)

    # compute steps from duration and hz
    speed_hz = max(1.0, float(speed_hz))
    total_steps = max(1, int(round(duration_sec * speed_hz)))

    # scale seed for the model (model was trained on scaled data)
    try:
        seed_scaled = scaler.transform(np.array(seed_original))
    except Exception:
        # if scaler fails, just use seed as-is
        seed_scaled = np.array(seed_original, dtype=np.float32)

    # Try model prediction; on failure use fallback procedural path
    try:
        preds = predict_sequence(model, scaler, seed_scaled.tolist(), pred_len=total_steps)
        # preds now in ORIGINAL scale (list of [x,y])
        if len(preds) < total_steps:
            # pad if weirdly short
            while len(preds) < total_steps:
                preds.append(preds[-1])
    except Exception:
        preds = _fallback_procedural_path(seed_original, total_steps, jitter_px=jitter_px)

    print(f"🚀 Live Path Tracing started for {duration_sec:.2f}s ({total_steps} steps). Press Ctrl+C to stop.")
    delay = 1.0 / speed_hz
    t0 = time.time()

    try:
        # step through predictions with timing control
        for (x_raw, y_raw) in preds:
            # micro-jitter
            x = x_raw + random.uniform(-jitter_px, jitter_px)
            y = y_raw + random.uniform(-jitter_px, jitter_px)

            # clamp to screen avoiding corners
            mx, my = _clamp_to_screen(x, y, margin=margin_px)

            # instantly move a small step (duration 0 for responsiveness)
            pyautogui.moveTo(mx, my, duration=0)
            # cadence control
            t0 += delay
            sleep_left = t0 - time.time()
            if sleep_left > 0:
                time.sleep(sleep_left)

    except KeyboardInterrupt:
        print("🛑 Live Path Tracing stopped by user.")
        return

    print("✅ Live Path Tracing complete.")
