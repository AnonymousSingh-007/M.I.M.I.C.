# # #trying dynamic time
# # import sys
# # from rich.console import Console
# # from mimic import spoofer, model, visuals

# # console = Console()
# # recorded_duration = 0  # Tracks how long data was collected


# # def main():
# #     global recorded_duration
# #     visuals.intro_sequence()

# #     while True:
# #         console.print("\n[bold blue]📋 Main Menu:[/bold blue]")
# #         console.print("[cyan][1][/cyan] 🖱️ Collect mouse data")
# #         console.print("[cyan][2][/cyan] 🧠 Train model")
# #         console.print("[cyan][3][/cyan] 🤖 Run cursor spoofer")
# #         console.print("[cyan][4][/cyan] ❌ Exit\n")

# #         choice = input("Choose an option: ").strip()

# #         # 1️⃣ Data collection
# #         if choice == "1":
# #             duration = input("Duration in seconds to collect? [default 15]: ").strip()
# #             try:
# #                 duration = int(duration)
# #             except ValueError:
# #                 duration = 15

# #             recorded_duration = duration
# #             spoofer.record_mouse_movements(duration=duration)

# #         # 2️⃣ Training
# #         elif choice == "2":
# #             csv_path = input("Path to CSV file (default: data/mouse_movements.csv): ").strip()
# #             if not csv_path:
# #                 csv_path = "data/mouse_movements.csv"

# #             trained_model = model.train(csv_path)
# #             if trained_model:
# #                 console.print(f"[green]✅ Model trained successfully from {csv_path}[/green]")

# #         # 3️⃣ Spoofing
# #         elif choice == "3":
# #             if recorded_duration == 0:
# #                 console.print("[bold red]⚠️ Please collect mouse data first![/bold red]")
# #                 continue

# #             max_spoof = int(0.75 * recorded_duration)
# #             default_spoof = min(5, max_spoof)
# #             prompt = f"Enter spoofing duration (max {max_spoof}s) [default {default_spoof}]: "
# #             spoof_duration = input(prompt).strip()

# #             try:
# #                 spoof_duration = int(spoof_duration)
# #                 if spoof_duration > max_spoof:
# #                     console.print(f"[yellow]⚠️ Duration capped to {max_spoof}s[/yellow]")
# #                     spoof_duration = max_spoof
# #             except ValueError:
# #                 spoof_duration = default_spoof

# #             try:
# #                 mdl = model.load_model()
# #                 spoofer.simulate_movement(mdl, duration=spoof_duration)
# #             except FileNotFoundError:
# #                 console.print("[bold red]❌ No trained model found! Please train the model first.[/bold red]")

# #         # 4️⃣ Exit
# #         elif choice == "4":
# #             console.print("[bold red]👋 Exiting M.I.M.I.C.[/bold red]")
# #             sys.exit()

# #         else:
# #             console.print("[bold red]❌ Invalid option. Please choose 1–4.[/bold red]")


# # if __name__ == "__main__":
# #     main()

# #changing to lstm# main.py
# import os
# from rich.console import Console
# from rich.table import Table
# from mimic import spoofer, model

# console = Console()

# def show_menu():
#     table = Table(title="🎮 MIMIC Control Panel")
#     table.add_column("Option", style="cyan", no_wrap=True)
#     table.add_column("Action", style="magenta")

#     table.add_row("1", "🖱️ Collect mouse data")
#     table.add_row("2", "🧠 Train LSTM model")
#     table.add_row("3", "🤖 Run cursor spoofer with Live Path Tracing")
#     table.add_row("4", "❌ Exit")

#     console.print(table)


# def main():
#     while True:
#         show_menu()
#         choice = input("Enter choice: ")

#         if choice == "1":
#             csv_path = "data/mouse_movements.csv"
#             os.makedirs("data", exist_ok=True)
#             spoofer.collect_data(csv_path, duration=30)

#         elif choice == "2":
#             path = input("Path to CSV file (default: data/mouse_movements.csv): ") or "data/mouse_movements.csv"
#             console.print(f"🧠 Training LSTM model on {path}")
#             model.train_lstm(path, model_path="models/mimic_lstm.pt", epochs=150, seq_len=30)
#             console.print("🎉 Training complete!")

#         elif choice == "3":
#             spoofer.live_path_tracing()
#         elif choice == "4":
#             print("Exiting... Goodbye!")
#             break
#         else:
#             print("❌ Invalid choice, try again.")


# if __name__ == "__main__":
#     main()

#lstm + game
import os
import sys
import time
from rich.console import Console
from rich.table import Table
from mimic import spoofer, model

console = Console()


def show_menu():
    table = Table(title="🎮 MIMIC Control Panel")
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Action", style="magenta")

    table.add_row("1", "🖱️ Collect mouse data")
    table.add_row("2", "🧠 Train LSTM model")
    table.add_row("3", "🤖 Run cursor spoofer with Live Path Tracing")
    table.add_row("4", "❌ Exit")

    console.print(table)


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    while True:
        show_menu()
        choice = input("Enter choice: ").strip()

        if choice == "1":
            csv_path = "data/mouse_movements.csv"
            try:
                secs = input("How many seconds to record? (default 30): ").strip()
                duration = int(secs) if secs.isdigit() else 30
                console.print(f"📡 Collecting mouse movement data for {duration} seconds -> {csv_path}")

                # Directly call spoofer (it handles progress & ESC stop)
                spoofer.collect_data(csv_path, duration=duration)

            except Exception as e:
                console.print(f"[red]❌ Error during recording: {e}")

        elif choice == "2":
            path = input("Path to CSV file (default: data/mouse_movements.csv): ").strip() or "data/mouse_movements.csv"
            if not os.path.exists(path):
                console.print(f"[red]❌ Data file {path} not found. Record first!")
                continue
            console.print(f"🧠 Training LSTM model on {path}")
            try:
                model.train_lstm(path, model_path="models/mimic_lstm.pt", epochs=150, seq_len=30)
                console.print("[green]🎉 Training complete!")
            except Exception as e:
                console.print(f"[red]❌ Error during training: {e}")

        elif choice == "3":
            if not os.path.exists("models/mimic_lstm.pt"):
                console.print("[red]❌ No trained model found. Train one first (option 2).")
                continue
            if not os.path.exists("data/mouse_movements.csv"):
                console.print("[red]❌ No data file found. Record some first (option 1).")
                continue

            recorded = spoofer.get_recorded_duration("data/mouse_movements.csv")
            if recorded is None:
                console.print("[yellow]⚠️ Could not determine recorded duration, defaulting to 10s.")
                recorded = 10.0

            console.print("[cyan]🤖 Running cursor spoofer with Live Path Tracing...")
            try:
                spoofer.live_path_tracing(
                    model_path="models/mimic_lstm.pt",
                    csv_path="data/mouse_movements.csv",
                    seq_len=30,
                    duration_sec=recorded,
                    speed_hz=60.0,
                    jitter_px=0.6,
                )
            except Exception as e:
                console.print(f"[red]❌ Error during spoofing: {e}")

        elif choice == "4":
            console.print("[green]Exiting... Goodbye!")
            sys.exit(0)

        else:
            console.print("[red]❌ Invalid choice, try again.")


if __name__ == "__main__":
    main()
