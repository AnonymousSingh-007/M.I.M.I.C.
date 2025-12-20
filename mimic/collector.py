import time
import pyautogui
import pandas as pd
import numpy as np
from pathlib import Path
from rich.progress import track

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


def collect_movements(filename, duration=120, sample_rate=125):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    data = []
    interval = 1.0 / sample_rate
    last_pos = None

    for _ in track(range(duration * sample_rate), description="Collecting"):
        start = time.perf_counter()
        x, y = pyautogui.position()
        t = time.time()

        if last_pos is not None:
            if np.hypot(x - last_pos[0], y - last_pos[1]) < 0.5:
                x, y = last_pos

        data.append((t, x, y))
        last_pos = (x, y)

        elapsed = time.perf_counter() - start
        time.sleep(max(0, interval - elapsed))

    df = pd.DataFrame(data, columns=["time", "x", "y"])

    dx = df["x"].diff().fillna(0)
    dy = df["y"].diff().fillna(0)
    dt = df["time"].diff().fillna(interval).clip(1e-4)
    speed = np.sqrt(dx ** 2 + dy ** 2) / dt

    out = pd.DataFrame({
        "x": df["x"],
        "y": df["y"],
        "dx": dx,
        "dy": dy,
        "speed": speed,
        "dt": dt
    })

    out.to_csv(filename, index=False)
