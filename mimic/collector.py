#curvature
# mimic/collector.py

import pandas as pd
import pyautogui
import time
import os
import numpy as np
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn

console = Console()

def collect_movements(filename="data/mouse_data.csv", duration=60):
    """
    Collects mouse cursor positions, computes features, and saves to CSV.
    Features: (dx, dy, speed, dt, curvature)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    console.print(f"\n[bold green]M.I.M.I.C. Collector Initialized[/bold green]")
    console.print(f"[cyan]Recording for[/cyan] {duration} [cyan]seconds...[/cyan]")
    console.print("[yellow]Start moving your mouse naturally.[/yellow]\n")

    time.sleep(2)  # small delay before starting
    
    raw_data = []
    sample_rate = 100  # Hz
    interval = 1.0 / sample_rate

    with Progress(
        TextColumn("[bold green]Recording Mouse Movements...[/bold green]"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Collecting...", total=duration)
        start_time = time.time()
        
        while not progress.finished:
            now = time.time()
            x, y = pyautogui.position()
            raw_data.append((now, x, y))
            time.sleep(interval)
            
            elapsed = time.time() - start_time
            progress.update(task, completed=elapsed)

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
    
    # Calculate curvature (angle of turn)
    curvature = np.zeros_like(speed)
    if len(deltas) > 1:
        v1 = deltas[:-1]
        v2 = deltas[1:]
        
        dot_product = np.einsum('ij,ij->i', v1, v2)
        mags = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        
        mags = np.where(mags == 0, 1e-8, mags)
        cos_angle = np.clip(dot_product / mags, -1.0, 1.0)
        
        curvature[1:] = np.arccos(cos_angle).reshape(-1, 1)

    # Create the features DataFrame
    features_df = pd.DataFrame({
        'time': times[1:],
        'x': coords[1:, 0],
        'y': coords[1:, 1],
        'dx': deltas[:, 0],
        'dy': deltas[:, 1],
        'speed': speed[:, 0],
        'dt': dts,
        'curvature': curvature[:, 0]
    })
    
    features_df.to_csv(filename, index=False)
    
    console.print(f"\n[bold green]✔ Data saved to[/bold green] [white]{filename}[/white]")
