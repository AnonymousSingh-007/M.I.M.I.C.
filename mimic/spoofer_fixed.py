import torch
import pyautogui
import numpy as np
import time
import joblib

from mimic.model import ImprovedLSTM


@torch.no_grad()
def spoof_realistic(
    model_path="models/lstm_fixed.pt",
    scaler_path="models/scaler_fixed.pkl",
    duration=3.5,
    steps=350,
    seq_len=200,
    horizon=10
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ── Load model & scaler ─────────────────────────────
    model = ImprovedLSTM(horizon=horizon).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    scaler = joblib.load(scaler_path)

    # ── User positions ─────────────────────────────────
    input("Cursor at START → ENTER")
    start = np.array(pyautogui.position(), dtype=np.float32)

    input("Cursor at TARGET → ENTER")
    target = np.array(pyautogui.position(), dtype=np.float32)

    print(f"Start: {start}, Target: {target}")
    print(f"Executing {steps} movement steps")

    screen_w, screen_h = pyautogui.size()

    # ── Initialize rolling sequence buffer ─────────────
    # Features: [x_norm, y_norm, dx, dy, speed, dt]
    seq = np.zeros((seq_len, 6), dtype=np.float32)

    seq[:, 0] = start[0] / screen_w
    seq[:, 1] = start[1] / screen_h

    pos = start.copy()
    prev_pos = pos.copy()

    dt = duration / steps

    # ── Execute motion ─────────────────────────────────
    for step in range(steps):
        # Scale sequence
        seq_scaled = scaler.transform(seq)
        x_tensor = torch.from_numpy(seq_scaled).unsqueeze(0).to(device)

        # Predict future dx, dy
        pred = model(x_tensor)[0].cpu().numpy()
        dx, dy = pred[:2]  # first predicted step only

        # Soft attraction toward target (NOT override)
        to_target = target - pos
        dist = np.linalg.norm(to_target) + 1e-6
        attraction = to_target / dist * min(dist, 8.0)

        # Combine LSTM motion + gentle guidance
        dx = dx * 0.85 + attraction[0] * 0.15
        dy = dy * 0.85 + attraction[1] * 0.15

        # Update position
        pos = pos + np.array([dx, dy], dtype=np.float32)

        # Clamp to screen
        pos[0] = np.clip(pos[0], 0, screen_w - 1)
        pos[1] = np.clip(pos[1], 0, screen_h - 1)

        # Compute derived features
        delta = pos - prev_pos
        speed = np.linalg.norm(delta) / max(dt, 1e-4)

        new_feat = np.array([
            pos[0] / screen_w,
            pos[1] / screen_h,
            delta[0],
            delta[1],
            speed,
            dt
        ], dtype=np.float32)

        # Roll sequence buffer
        seq[:-1] = seq[1:]
        seq[-1] = new_feat

        # Move cursor
        pyautogui.moveTo(int(pos[0]), int(pos[1]))
        time.sleep(dt)

        prev_pos = pos.copy()

    print("✔ LSTM spoofing completed")
