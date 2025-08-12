# # import time
# # import csv
# # import pyautogui
# # import pandas as pd
# # from pynput import mouse
# # from rich.console import Console
# # from mimic.model import predict  # predict(model, t, t0)

# # console = Console()


# # def record_mouse_movements(duration=15, output_file="data/mouse_movements.csv"):
# #     """
# #     Records mouse movements for the given duration and saves to a CSV.
# #     """
# #     console.print(f"[bold green]🎯 Recording mouse movements for {duration} seconds...[/bold green]")
# #     movements = []

# #     def on_move(x, y):
# #         t = time.time()
# #         movements.append((t, x, y))

# #     listener = mouse.Listener(on_move=on_move)
# #     listener.start()

# #     time.sleep(duration)
# #     listener.stop()

# #     # Save to CSV
# #     with open(output_file, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow(["time", "x", "y"])
# #         writer.writerows(movements)

# #     console.print(f"[green]✅ Data saved to [bold]{output_file}[/bold][/green]")


# # def simulate_movement(model, duration=10, interval=0.01):
# #     """
# #     Uses a trained model to spoof cursor movement for a given duration.
# #     """
# #     console.print(f"[bold magenta]⏱️ Spoofing for {duration} seconds...[/bold magenta]")
# #     time.sleep(1)
# #     console.print("[yellow]🖱️ Releasing control to M.I.M.I.C...[/yellow]")

# #     t0 = time.time()
# #     while time.time() - t0 < duration:
# #         t = time.time()
# #         try:
# #             x, y = predict(model, t, t0)
# #             pyautogui.moveTo(int(x), int(y), duration=interval)
# #         except Exception as e:
# #             console.print(f"[red]⚠️ Movement failed: {e}[/red]")
# #             break
# #         time.sleep(interval)

# #     console.print("[green]✔ Spoofing complete.[/green]")

# #trying dynamic time
# import time
# import csv
# import pyautogui
# import pandas as pd
# from pynput import mouse
# from rich.console import Console
# from mimic.model import predict  # predict(model, t, t0)
# import os

# console = Console()


# # def record_mouse_movements(duration=15, output_file="data/mouse_movements.csv"):
# #     """
# #     Records mouse movements for the given duration and saves to a CSV.
# #     """
# #     console.print(f"[bold green]🎯 Recording mouse movements for {duration} seconds...[/bold green]")
# #     movements = []

# #     def on_move(x, y):
# #         t = time.time()
# #         movements.append((t, x, y))

# #     listener = mouse.Listener(on_move=on_move)
# #     listener.start()

# #     time.sleep(duration)
# #     listener.stop()

# #     # Save to CSV
# #     with open(output_file, "w", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow(["timestamp", "x", "y"])
# #         writer.writerows(movements)

# #     console.print(f"[green]✅ Data saved to [bold]{output_file}[/bold][/green]")

# #speed tracking

# def record_mouse_movements(duration=15, save_path="data/mouse_movements.csv"):
#     print(f"🕵️ Collecting mouse data for {duration} seconds... Move your mouse naturally.")
#     start_time = time.time()
#     last_time = start_time
#     last_pos = pyautogui.position()

#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     with open(save_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["time", "x", "y", "dx", "dy", "speed", "pause"])

#         while (time.time() - start_time) < duration:
#             current_time = time.time()
#             x, y = pyautogui.position()
#             elapsed = current_time - start_time
#             delta_time = current_time - last_time

#             dx = x - last_pos[0]
#             dy = y - last_pos[1]
#             distance = (dx**2 + dy**2) ** 0.5
#             speed = distance / delta_time if delta_time > 0 else 0
#             pause = 1 if distance < 2 else 0  # Small movement = idle

#             writer.writerow([elapsed, x, y, dx, dy, speed, pause])

#             last_pos = (x, y)
#             last_time = current_time
#             time.sleep(0.01)  # 100Hz sampling

#     print(f"✅ Data collection complete. Saved to: {save_path}")


# def simulate_movement(model, duration=5, interval=0.01):
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

#Trying Dynamic and idle integration
import time
import csv
import os
import random
import pyautogui
from pathlib import Path
from pynput import mouse
from rich.console import Console
import torch
import math

console = Console()

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "mouse_movements.csv")


# 🎯 Record mouse movements
def record_mouse_movements(duration=15):
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "dx", "dy", "speed", "pause", "x", "y"])

        start_time = time.time()
        prev_time = start_time
        prev_x, prev_y = pyautogui.position()

        def on_move(x, y):
            nonlocal prev_time, prev_x, prev_y
            now = time.time()
            if now - start_time > duration:
                return False

            dx = x - prev_x
            dy = y - prev_y
            dt = now - prev_time if now - prev_time > 0 else 1e-6
            speed = math.sqrt(dx**2 + dy**2) / dt
            pause = 1 if dx == 0 and dy == 0 else 0

            rel_time = now - start_time
            writer.writerow([rel_time, dx, dy, speed, pause, x, y])

            prev_x, prev_y = x, y
            prev_time = now
            return True

        console.print(f"🎙️ Recording mouse for {duration} seconds...")
        with mouse.Listener(on_move=on_move) as listener:
            listener.join()

    console.print(f"✅ Data saved to: {DATA_FILE}")


# 🤖 Simulate cursor movement
def simulate_movement(model, duration=5):
    console.print(f"🌀 Spoofing for {duration} seconds...")
    console.print("🖱️ Releasing control to M.I.M.I.C...")

    pyautogui.FAILSAFE = False  # Prevent failsafe crash
    start_time = time.time()
    prev_time = start_time
    prev_x, prev_y = pyautogui.position()

    screen_width, screen_height = pyautogui.size()
    margin = 5  # Safe margin from edges

    while time.time() - start_time < duration:
        now = time.time()
        rel_time = now - start_time

        dx = prev_x - prev_x  # initially 0
        dy = prev_y - prev_y
        dt = now - prev_time if now - prev_time > 0 else 1e-6
        speed = math.sqrt(dx**2 + dy**2) / dt
        pause = 1 if dx == 0 and dy == 0 else 0

        # Build input vector
        input_vector = torch.tensor([[rel_time, dx, dy, speed, pause]], dtype=torch.float32)

        # Normalize
        input_vector = (input_vector - torch.tensor(model.mean)) / torch.tensor(model.std)

        # Predict next position
        with torch.no_grad():
            pred_x, pred_y = model(input_vector).squeeze().tolist()

        # Clamp inside safe screen area
        pred_x = max(margin, min(screen_width - margin, pred_x))
        pred_y = max(margin, min(screen_height - margin, pred_y))

        pyautogui.moveTo(pred_x, pred_y, duration=random.uniform(0.02, 0.1))

        prev_x, prev_y = pred_x, pred_y
        prev_time = now
