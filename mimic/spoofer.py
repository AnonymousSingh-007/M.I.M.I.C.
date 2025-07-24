import time
import pyautogui
import numpy as np
from rich.console import Console

console = Console()

def simulate_movement(model, duration=5, interval=0.01):
    """
    Uses the trained model to simulate mouse movement.

    Args:
        model: Trained regression model.
        duration: How many seconds to spoof for.
        interval: Time between steps (seconds).
    """
    console.print(f"[bold magenta]⏱️ Spoofing for {duration} seconds...[/bold magenta]")
    time.sleep(2)
    console.print("[yellow]🖱️ Releasing control to M.I.M.I.C...[/yellow]")

    start_time = time.time()

    while time.time() - start_time < duration:
        current_time = np.array([[time.time()]])
        predicted_pos = model.predict(current_time)[0]
        x, y = int(predicted_pos[0]), int(predicted_pos[1])

        try:
            pyautogui.moveTo(x, y, duration=interval)
        except Exception as e:
            console.print(f"[red]Movement error:[/red] {e}")
            break

        time.sleep(interval)
    
    console.print("[green]✔ Spoofing complete.[/green]")
