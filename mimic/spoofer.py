# import time
# import csv
# import pyautogui
# import pandas as pd
# from pynput import mouse
# from rich.console import Console
# from mimic.model import predict  # predict(model, t, t0)

# console = Console()


# def record_mouse_movements(duration=15, output_file="data/mouse_movements.csv"):
#     """
#     Records mouse movements for the given duration and saves to a CSV.
#     """
#     console.print(f"[bold green]🎯 Recording mouse movements for {duration} seconds...[/bold green]")
#     movements = []

#     def on_move(x, y):
#         t = time.time()
#         movements.append((t, x, y))

#     listener = mouse.Listener(on_move=on_move)
#     listener.start()

#     time.sleep(duration)
#     listener.stop()

#     # Save to CSV
#     with open(output_file, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(["time", "x", "y"])
#         writer.writerows(movements)

#     console.print(f"[green]✅ Data saved to [bold]{output_file}[/bold][/green]")


# def simulate_movement(model, duration=10, interval=0.01):
#     """
#     Uses a trained model to spoof cursor movement for a given duration.
#     """
#     console.print(f"[bold magenta]⏱️ Spoofing for {duration} seconds...[/bold magenta]")
#     time.sleep(1)
#     console.print("[yellow]🖱️ Releasing control to M.I.M.I.C...[/yellow]")

#     t0 = time.time()
#     while time.time() - t0 < duration:
#         t = time.time()
#         try:
#             x, y = predict(model, t, t0)
#             pyautogui.moveTo(int(x), int(y), duration=interval)
#         except Exception as e:
#             console.print(f"[red]⚠️ Movement failed: {e}[/red]")
#             break
#         time.sleep(interval)

#     console.print("[green]✔ Spoofing complete.[/green]")

#trying dynamic time
import time
import csv
import pyautogui
import pandas as pd
from pynput import mouse
from rich.console import Console
from mimic.model import predict  # predict(model, t, t0)

console = Console()


def record_mouse_movements(duration=15, output_file="data/mouse_movements.csv"):
    """
    Records mouse movements for the given duration and saves to a CSV.
    """
    console.print(f"[bold green]🎯 Recording mouse movements for {duration} seconds...[/bold green]")
    movements = []

    def on_move(x, y):
        t = time.time()
        movements.append((t, x, y))

    listener = mouse.Listener(on_move=on_move)
    listener.start()

    time.sleep(duration)
    listener.stop()

    # Save to CSV
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y"])
        writer.writerows(movements)

    console.print(f"[green]✅ Data saved to [bold]{output_file}[/bold][/green]")


def simulate_movement(model, duration=5, interval=0.01):
    """
    Uses a trained model to spoof cursor movement for a given duration.
    """
    console.print(f"[bold magenta]⏱️ Spoofing for {duration} seconds...[/bold magenta]")
    time.sleep(1)
    console.print("[yellow]🖱️ Releasing control to M.I.M.I.C...[/yellow]")

    t0 = time.time()
    while time.time() - t0 < duration:
        t = time.time()
        try:
            x, y = predict(model, t, t0)
            pyautogui.moveTo(int(x), int(y), duration=interval)
        except Exception as e:
            console.print(f"[red]⚠️ Movement failed: {e}[/red]")
            break
        time.sleep(interval)

    console.print("[green]✔ Spoofing complete.[/green]")
