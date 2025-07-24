import pyautogui
import time
import csv
import os
from rich.console import Console
from rich.progress import track

console = Console()

def collect_movements(filename="data/mouse_movements.csv", duration=10):
    """
    Collects mouse cursor positions over a given duration and saves to CSV.
    
    Args:
        filename (str): Path to the CSV file where data will be saved.
        duration (int): Duration of collection in seconds.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    console.print(f"\n[bold green]M.I.M.I.C. Collector Initialized[/bold green]")
    console.print(f"[cyan]Recording for[/cyan] {duration} [cyan]seconds...[/cyan]")
    console.print("[yellow]Start moving your mouse naturally.[/yellow]\n")

    time.sleep(2)  # small delay before starting
    start = time.time()

    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'x', 'y'])

        for _ in track(range(int(duration * 100)), description="Collecting..."):
            now = time.time()
            x, y = pyautogui.position()
            writer.writerow([now, x, y])
            time.sleep(0.01)  # 100Hz sample rate (10ms)

    console.print(f"\n[bold green]✔ Data saved to[/bold green] [white]{filename}[/white]")
