#mimic/collector.py

import pandas as pd
import pyautogui
import time
import os
import numpy as np
from rich.console import Console
from rich.progress import track

console = Console()

def collect_movements(filename="data/mouse_data.csv", duration=30):
    """
    Collects mouse cursor positions, computes features, and saves to CSV.
    Features: (dx, dy, speed, dt)
    
    Args:
        filename (str): Path to the CSV file where data will be saved.
        duration (int): Duration of collection in seconds.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    console.print(f"\n[bold green]M.I.M.I.C. Collector Initialized[/bold green]")
    console.print(f"[cyan]Recording for[/cyan] {duration} [cyan]seconds...[/cyan]")
    console.print("[yellow]Start moving your mouse naturally.[/yellow]\n")

    time.sleep(2)  # small delay before starting
    
    raw_data = []
    
    # Use a high sample rate for accurate deltas
    sample_rate = 100 # Hz
    interval = 1.0 / sample_rate

    for _ in track(range(int(duration * sample_rate)), description="Collecting..."):
        now = time.time()
        x, y = pyautogui.position()
        raw_data.append((now, x, y))
        time.sleep(interval)

    if not raw_data:
        console.print("[red]❌ No data was collected. Please try again.[/red]")
        return
        
    df = pd.DataFrame(raw_data, columns=['time', 'x', 'y'])
    
    # Compute features
    coords = df[['x', 'y']].values.astype(np.float32)
    deltas = np.diff(coords, axis=0)
    times = df['time'].to_numpy(dtype=np.float32)
    dts = times[1:] - times[:-1]
    dts = np.where(dts <= 0, 1e-3, dts)  # avoid zero dt
    
    speed = np.linalg.norm(deltas, axis=1, keepdims=True)
    
    # Create the features DataFrame
    features_df = pd.DataFrame({
        'time': times[1:],
        'x': coords[1:, 0],
        'y': coords[1:, 1],
        'dx': deltas[:, 0],
        'dy': deltas[:, 1],
        'speed': speed[:, 0],
        'dt': dts
    })
    
    features_df.to_csv(filename, index=False)
    
    console.print(f"\n[bold green]✔ Data saved to[/bold green] [white]{filename}[/white]")
