#curvature
# mimic/collector.py

# import pandas as pd
# import pyautogui
# import time
# import os
# import numpy as np
# from rich.console import Console
# from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn

# console = Console()

# def collect_movements(filename="data/mouse_data.csv", duration=60):
#     """
#     Collects mouse cursor positions, computes features, and saves to CSV.
#     Features: (dx, dy, speed, dt, curvature)
#     """
#     os.makedirs(os.path.dirname(filename), exist_ok=True)

#     console.print(f"\n[bold green]M.I.M.I.C. Collector Initialized[/bold green]")
#     console.print(f"[cyan]Recording for[/cyan] {duration} [cyan]seconds...[/cyan]")
#     console.print("[yellow]Start moving your mouse naturally.[/yellow]\n")

#     time.sleep(2)  # small delay before starting
    
#     raw_data = []
#     sample_rate = 200  # Hz
#     interval = 1.0 / sample_rate

#     with Progress(
#         TextColumn("[bold green]Recording Mouse Movements...[/bold green]"),
#         BarColumn(bar_width=None),
#         "[progress.percentage]{task.percentage:>3.0f}%",
#         TimeRemainingColumn(),
#         console=console,
#         transient=True
#     ) as progress:
#         task = progress.add_task("Collecting...", total=duration)
#         start_time = time.time()
        
#         while not progress.finished:
#             now = time.time()
#             x, y = pyautogui.position()
#             raw_data.append((now, x, y))
#             time.sleep(interval)
            
#             elapsed = time.time() - start_time
#             progress.update(task, completed=elapsed)

#     if not raw_data:
#         console.print("[red]❌ No data was collected. Please try again.[/red]")
#         return
        
#     df = pd.DataFrame(raw_data, columns=['time', 'x', 'y'])
    
#     # Compute features
#     coords = df[['x', 'y']].values.astype(np.float32)
#     deltas = np.diff(coords, axis=0)
#     times = df['time'].to_numpy(dtype=np.float32)
#     dts = times[1:] - times[:-1]
#     dts = np.where(dts <= 0, 1e-3, dts)  # avoid zero dt
    
#     speed = np.linalg.norm(deltas, axis=1, keepdims=True)
    
#     # Calculate curvature (angle of turn)
#     curvature = np.zeros_like(speed)
#     if len(deltas) > 1:
#         v1 = deltas[:-1]
#         v2 = deltas[1:]
        
#         dot_product = np.einsum('ij,ij->i', v1, v2)
#         mags = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        
#         mags = np.where(mags == 0, 1e-8, mags)
#         cos_angle = np.clip(dot_product / mags, -1.0, 1.0)
        
#         curvature[1:] = np.arccos(cos_angle).reshape(-1, 1)

#     # Create the features DataFrame
#     features_df = pd.DataFrame({
#         'time': times[1:],
#         'x': coords[1:, 0],
#         'y': coords[1:, 1],
#         'dx': deltas[:, 0],
#         'dy': deltas[:, 1],
#         'speed': speed[:, 0],
#         'dt': dts,
#         'curvature': curvature[:, 0]
#     })
    
#     features_df.to_csv(filename, index=False)
    
#     console.print(f"\n[bold green]✔ Data saved to[/bold green] [white]{filename}[/white]")

#testing seq2seq
# mimic/collector.py

# import time
# import pyautogui
# import pandas as pd
# import numpy as np
# from rich.console import Console

# console = Console()


# def compute_curvature(dx1, dy1, dx2, dy2):
#     """Approximate curvature from two consecutive motion vectors."""
#     v1 = np.array([dx1, dy1])
#     v2 = np.array([dx2, dy2])
#     if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
#         return 0.0
#     cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#     cos_theta = np.clip(cos_theta, -1.0, 1.0)
#     return np.arccos(cos_theta)


# def collect_movements(output_csv="data/mouse_data.csv", duration=60):
#     pyautogui.FAILSAFE = False
#     console.print(f"🎯 Collecting mouse data for {duration} seconds...")

#     data = []
#     start_time = time.time()
#     prev_x, prev_y = pyautogui.position()
#     prev_time = start_time
#     prev_dx, prev_dy = 0, 0

#     screen_width, screen_height = pyautogui.size()

#     while time.time() - start_time < duration:
#         x, y = pyautogui.position()
#         now = time.time()

#         dx, dy = x - prev_x, y - prev_y
#         dt = now - prev_time
#         speed = np.hypot(dx, dy) / dt if dt > 0 else 0.0
#         curvature = compute_curvature(prev_dx, prev_dy, dx, dy)

#         x_norm = x / screen_width
#         y_norm = y / screen_height

#         data.append([dx, dy, speed, dt, curvature, x_norm, y_norm])

#         prev_x, prev_y, prev_time = x, y, now
#         prev_dx, prev_dy = dx, dy

#         time.sleep(0.01)

#     df = pd.DataFrame(
#         data, columns=["dx", "dy", "speed", "dt", "curvature", "x_norm", "y_norm"]
#     )
#     df.to_csv(output_csv, index=False)
#     console.print(f"✅ Saved {len(df)} samples to {output_csv}")
#     return df


#gemini appraoch
# # mimic/collector.py

# import time
# import pyautogui
# import pandas as pd
# import numpy as np
# from rich.console import Console

# console = Console()


# def compute_curvature(dx1, dy1, dx2, dy2):
#     v1, v2 = np.array([dx1, dy1]), np.array([dx2, dy2])
#     if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
#         return 0.0
#     cos_theta = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
#     return np.arccos(cos_theta)


# def collect_movements(output_csv="data/mouse_data.csv", duration=60, sample_rate=100):
#     """Record mouse deltas + features"""
#     pyautogui.FAILSAFE = False
#     console.print(f"🎯 Collecting {duration}s @ {sample_rate}Hz...")

#     data, start_time = [], time.time()
#     prev_x, prev_y = pyautogui.position()
#     prev_time, prev_dx, prev_dy, prev_speed = start_time, 0.0, 0.0, 0.0

#     target_interval = 1.0 / sample_rate
#     sw, sh = pyautogui.size()

#     while time.time() - start_time < duration:
#         loop_start = time.time()
#         x, y = pyautogui.position()
#         now = time.time()

#         if (x, y) != (prev_x, prev_y):
#             dx, dy, dt = x - prev_x, y - prev_y, now - prev_time
#             speed = np.hypot(dx, dy) / dt if dt > 0 else 0.0
#             accel = (speed - prev_speed) / dt if dt > 0 else 0.0
#             curvature = compute_curvature(prev_dx, prev_dy, dx, dy)

#             data.append([
#                 dx / sw, dy / sh, speed, dt, curvature,
#                 accel / sw, accel / sh
#             ])

#             prev_x, prev_y, prev_dx, prev_dy, prev_speed = x, y, dx, dy, speed
#         prev_time = now

#         sleep_dur = target_interval - (time.time() - loop_start)
#         if sleep_dur > 0:
#             time.sleep(sleep_dur)

#     df = pd.DataFrame(data, columns=[
#         "dx_norm", "dy_norm", "speed", "dt", "curvature", "x_accel_norm", "y_accel_norm"
#     ])
#     df.to_csv(output_csv, index=False)
#     console.print(f"✅ Saved {len(df)} samples → {output_csv}")
#     return df











#mimic/collector.py - E


# mimic/collector.py
# mimic/collector.py
# mimic/collector.py
import pandas as pd
import pyautogui
import time
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()

def collect_movements(filename="data/mouse_data.csv", duration=30):
    console.print("[yellow]Start moving your mouse naturally.[/yellow]\n")
    time.sleep(1)
    raw_data = []
    sample_rate = 100
    interval = 1.0 / sample_rate

    for _ in track(range(int(duration * sample_rate)), description="Collecting..."):
        now = time.time()
        x, y = pyautogui.position()
        raw_data.append((now, x, y))
        time.sleep(interval)

    if not raw_data:
        console.print("[red]❌ No data collected. Try again.")
        return

    df = pd.DataFrame(raw_data, columns=['time', 'x', 'y'])
    coords = df[['x', 'y']].values.astype(np.float32)
    deltas = np.diff(coords, axis=0)
    times = df['time'].to_numpy(dtype=np.float32)
    dts = times[1:] - times[:-1]
    dts = np.where(dts <= 0, 1e-3, dts)
    speed = np.linalg.norm(deltas, axis=1)

    features_df = pd.DataFrame({
        'time': times[1:],
        'x': coords[1:, 0],
        'y': coords[1:, 1],
        'dx': deltas[:, 0],
        'dy': deltas[:, 1],
        'speed': speed,
        'dt': dts
    })

    features_df.to_csv(filename, index=False)
    console.print(f"\n[bold green]✔ Data saved to[/bold green] [white]{filename}[/white]")
