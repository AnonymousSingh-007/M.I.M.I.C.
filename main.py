# # main.py
# #curvature
# import os
# import sys
# from rich.console import Console
# from rich.table import Table
# from mimic.collector import collect_movements
# from mimic.lstm_trainer import train_lstm
# from mimic.spoofer import spoof_and_plot
# from mimic.visuals import display_intro, display_status, display_success, display_error

# console = Console()

# CSV_PATH = "data/mouse_data.csv"
# MODEL_PATH = "models/mimic_lstm.pt"
# SCALER_PATH = "models/mimic_scaler.pkl"

# def show_menu():
#     table = Table(title="🎮 M.I.M.I.C. Control Panel")
#     table.add_column("Option", style="cyan", no_wrap=True)
#     table.add_column("Action", style="magenta")

#     table.add_row("1", "🖱️ Collect mouse data")
#     table.add_row("2", "🧠 Train LSTM model")
#     table.add_row("3", "🤖 Run cursor spoofer (Live + Graph)")
#     table.add_row("4", "❌ Exit")

#     console.print(table)

# def main():
#     display_intro()
#     os.makedirs("data", exist_ok=True)
#     os.makedirs("models", exist_ok=True)

#     while True:
#         show_menu()
#         choice = input("Enter choice: ").strip()

#         if choice == "1":
#             secs = input("How many seconds to record? (default 30): ").strip()
#             duration = int(secs) if secs.isdigit() else 30
#             display_status(f"Collecting mouse data for {duration} seconds")
#             collect_movements(CSV_PATH, duration=duration)
#             display_success(f"Data saved to {CSV_PATH}")

#         elif choice == "2":
#             if not os.path.exists(CSV_PATH):
#                 display_error("No data file found. Please record mouse data first (option 1).")
#                 continue

#             if os.path.exists(MODEL_PATH):
#                 overwrite = input("⚠️ A trained model already exists. Retrain? (y/n): ").strip().lower()
#                 if overwrite != "y":
#                     display_status("Training cancelled. Using existing model.")
#                     continue

#             display_status(f"Training LSTM model on {CSV_PATH}")
#             try:
#                 train_lstm(CSV_PATH, model_path=MODEL_PATH, scaler_path=SCALER_PATH)
#                 display_success("Training complete!")
#             except Exception as e:
#                 display_error(f"Error during training: {e}")

#         elif choice == "3":
#             if not os.path.exists(CSV_PATH):
#                 display_error("No CSV data found. Please record mouse data first (option 1).")
#                 continue
#             if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
#                 display_error("No trained model found. Please train one first (option 2).")
#                 continue

#             display_status("Running cursor spoofer... Graph will pop up automatically!")
#             try:
#                 spoof_and_plot(model_path=MODEL_PATH, scaler_path=SCALER_PATH, csv_path=CSV_PATH)
#                 display_success("Spoofing finished.")
#             except Exception as e:
#                 display_error(f"Error during spoofing: {e}")

#         elif choice == "4":
#             display_status("Exiting... Goodbye!")
#             sys.exit(0)

#         else:
#             display_error("Invalid choice, try again.")

# if __name__ == "__main__":
#     main()


# main.py
# import os
# import sys
# from rich.console import Console
# from rich.table import Table
# from mimic.collector import collect_movements
# from mimic.lstm_trainer import train_model
# from mimic.spoofer import spoof_and_collect
# from mimic.evaluator import plot_trajectories

# console = Console()

# CSV_PATH = "data/mouse_data.csv"
# MODEL_PATH = "models/mimic_model.pt"
# SCALER_PATH = "models/mimic_scaler.pkl"


# def show_menu():
#     table = Table(title="🎮 M.I.M.I.C. Control Panel")
#     table.add_column("Option", style="cyan")
#     table.add_column("Action", style="magenta")
#     table.add_row("1", "🖱️ Collect new mouse data (DELETES OLD DATA)")
#     table.add_row("2", "🧠 Train the model")
#     table.add_row("3", "🤖 Run cursor spoofer")
#     table.add_row("4", "❌ Exit")
#     console.print(table)


# def main():
#     os.makedirs("data", exist_ok=True)
#     os.makedirs("models", exist_ok=True)

#     config = {
#         "model_type": "seq2seq",
#         "epochs": 120,
#         "num_layers": 2,
#         "hidden_size": 128,
#         "pred_horizon": 15,
#         "seq_len": 100,
#     }
#     console.print("[bold cyan]Current Config:[/bold cyan]", config)

#     while True:
#         show_menu()
#         choice = input("Enter choice: ").strip()

#         if choice == "1":
#             if os.path.exists(CSV_PATH):
#                 try:
#                     os.remove(CSV_PATH)
#                     console.print(f"[yellow]Deleted old data file: {CSV_PATH}[/yellow]")
#                 except OSError as e:
#                     console.print(f"[red]Error deleting file: {e}[/red]")
#             duration = int(input("How many seconds to record? (e.g., 120): ").strip() or 120)
#             collect_movements(CSV_PATH, duration=duration)

#         elif choice == "2":
#             if not os.path.exists(CSV_PATH):
#                 console.print("[red]No data file found. Collect data first (option 1).[/red]")
#                 continue
#             train_model(CSV_PATH, MODEL_PATH, SCALER_PATH, **config)

#         elif choice == "3":
#             if not os.path.exists(MODEL_PATH):
#                 console.print("[red]No trained model found. Train first (option 2).[/red]")
#                 continue

#             spoofer_config = config.copy()
#             spoofer_config.pop("epochs", None)

#             recorded, generated = spoof_and_collect(
#                 MODEL_PATH, SCALER_PATH, CSV_PATH, **spoofer_config
#             )
#             if recorded is not None and generated is not None:
#                 plot_trajectories(recorded, generated)

#         elif choice == "4":
#             sys.exit(0)
#         else:
#             console.print("[yellow]Invalid choice. Try again.[/yellow]")


# if __name__ == "__main__":
#     main()


#gemini approach

# main.py
# # main.py

# import os
# import sys
# import json
# from rich.console import Console
# from rich.table import Table
# from mimic.collector import collect_movements
# from mimic.lstm_trainer import train_model
# from mimic.spoofer import spoof_and_collect
# from mimic.evaluator import plot_trajectories
# from mimic.visuals import display_intro, display_success, display_error

# console = Console()

# CSV_PATH = "data/mouse_data.csv"
# MODEL_PATH = "models/mimic_model.pt"
# SCALER_PATH = "models/mimic_scaler.pkl"
# CONFIG_PATH = "config.json"


# def show_menu():
#     """Display menu table"""
#     table = Table(title="🎮 M.I.M.I.C. Control Panel")
#     table.add_column("Option", style="cyan")
#     table.add_column("Action", style="magenta")
#     table.add_row("1", "🖱️ Collect new mouse data (DELETES OLD DATA)")
#     table.add_row("2", "🧠 Train the model")
#     table.add_row("3", "🤖 Run cursor spoofer")
#     table.add_row("4", "❌ Exit")
#     console.print(table)


# def load_config():
#     """Load config JSON or fallback to defaults"""
#     default_config = {
#         "model_type": "seq2seq",
#         "epochs": 120,
#         "num_layers": 2,
#         "hidden_size": 128,
#         "pred_horizon": 15,   # renamed consistently
#         "seq_len": 100,
#     }
#     if os.path.exists(CONFIG_PATH):
#         try:
#             with open(CONFIG_PATH, "r") as f:
#                 config = json.load(f)
#             console.print(f"[green]Loaded configuration from {CONFIG_PATH}[/green]")
#             return {**default_config, **config}  # merge with defaults
#         except json.JSONDecodeError:
#             console.print(f"[yellow]Invalid {CONFIG_PATH}. Using defaults.[/yellow]")
#             return default_config
#     else:
#         console.print(f"[yellow]No {CONFIG_PATH} found. Using defaults.[/yellow]")
#         return default_config


# def main():
#     os.makedirs("data", exist_ok=True)
#     os.makedirs("models", exist_ok=True)

#     display_intro()

#     config = load_config()
#     console.print("[bold cyan]Current Config:[/bold cyan]", config)

#     while True:
#         show_menu()
#         choice = input("Enter choice: ").strip()

#         if choice == "1":
#             if os.path.exists(CSV_PATH):
#                 confirm = input(f"Delete {CSV_PATH}? (y/N): ").strip().lower()
#                 if confirm == "y":
#                     os.remove(CSV_PATH)
#                     console.print(f"[yellow]Deleted old file {CSV_PATH}[/yellow]")
#                 else:
#                     continue
#             try:
#                 duration = int(input("Record duration in seconds (default 120): ") or "120")
#             except ValueError:
#                 console.print("[red]Invalid input. Enter a number.[/red]")
#                 continue
#             collect_movements(CSV_PATH, duration=duration)
#             display_success("Data collection complete!")

#         elif choice == "2":
#             if not os.path.exists(CSV_PATH):
#                 display_error("No data found. Collect data first.")
#                 continue
#             train_model(
#                 CSV_PATH, MODEL_PATH, SCALER_PATH,
#                 seq_len=config["seq_len"],
#                 pred_horizon=config["pred_horizon"],
#                 hidden_size=config["hidden_size"],
#                 num_layers=config["num_layers"],
#                 epochs=config["epochs"]
#             )
#             display_success("Model training complete!")

#         elif choice == "3":
#             if not os.path.exists(MODEL_PATH):
#                 display_error("No trained model. Train first (option 2).")
#                 continue
#             rec, gen = spoof_and_collect(
#                 MODEL_PATH, SCALER_PATH, CSV_PATH,
#                 seq_len=config["seq_len"],
#                 pred_horizon=config["pred_horizon"],
#                 hidden_size=config["hidden_size"],
#                 num_layers=config["num_layers"]
#             )
#             if rec is not None and gen is not None:
#                 plot_trajectories(rec, gen)
#             display_success("Spoofing and evaluation complete!")

#         elif choice == "4":
#             console.print("[bold green]Exiting. Goodbye![/bold green]")
#             sys.exit(0)
#         else:
#             console.print("[yellow]Invalid choice. Try again.[/yellow]")


# if __name__ == "__main__":
#     main()







#main.py - E

# main.py
#!/usr/bin/env python3
# main.py


# main.py
# main.py
import os
import sys
from mimic.collector import collect_movements
from mimic.lstm_trainer import train_lstm
from mimic.spoofer import spoof_and_plot
from mimic import visuals

CSV_PATH = "data/mouse_data.csv"
MODEL_PATH = "models/mimic_lstm.pt"
SCALER_PATH = "models/mimic_scaler.pkl"

def show_menu():
    visuals.display_intro()
    print("\n[1] 🖱️ Collect mouse data")
    print("[2] 🧠 Train LSTM model")
    print("[3] 🤖 Run cursor spoofer")
    print("[4] ❌ Exit")

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    while True:
        show_menu()
        choice = input("Enter choice: ").strip()

        if choice == "1":
            secs = input("How many seconds to record? (default 30): ").strip()
            duration = int(secs) if secs.isdigit() else 30
            visuals.display_status(f"Collecting mouse data for {duration} seconds...")
            collect_movements(CSV_PATH, duration=duration)
            visuals.display_success("Mouse data collected successfully!")
            visuals.show_achievement("Data Collector", f"Recorded {duration} seconds of mouse movement.")

        elif choice == "2":
            if not os.path.exists(CSV_PATH):
                visuals.display_error("No data file found. Record mouse data first.")
                continue

            overwrite = "y"
            if os.path.exists(MODEL_PATH):
                overwrite = input("A trained model exists. Retrain? (y/n): ").strip().lower()
            if overwrite != "y":
                visuals.display_status("Using existing model. Training skipped.")
                continue

            visuals.display_status("Training LSTM model...")
            visuals.show_progress_bar("Training LSTM", total=200, speed=0.03)
            try:
                train_lstm(
                    CSV_PATH,
                    model_path=MODEL_PATH,
                    scaler_path=SCALER_PATH,
                    hidden_size=256,  # must match saved model
                    epochs=120
                )
                visuals.display_success("Training complete!")
                visuals.show_achievement("AI Trainer", "LSTM model trained successfully.")
            except Exception as e:
                visuals.display_error(f"Error during training: {e}")

        elif choice == "3":
            if not os.path.exists(CSV_PATH):
                visuals.display_error("No CSV data found. Record mouse data first.")
                continue
            if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
                visuals.display_error("No trained model found. Train one first.")
                continue

            visuals.display_status("Running cursor spoofer...")
            try:
                spoof_and_plot(
                    model_path=MODEL_PATH,
                    scaler_path=SCALER_PATH,
                    csv_path=CSV_PATH,
                    seq_len=150,
                    steps=800,
                    move_delay=0.001
                )
                visuals.display_success("Spoofing finished successfully!")
                visuals.show_achievement("Cursor Phantom", "Generated realistic mouse movement.")
            except Exception as e:
                visuals.display_error(f"Error during spoofing: {e}")

        elif choice == "4":
            visuals.display_status("Exiting... Goodbye!")
            sys.exit(0)
        else:
            visuals.display_error("Invalid choice, try again.")

if __name__ == "__main__":
    main()
