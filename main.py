# main.py
#curvature
import os
import sys
from rich.console import Console
from rich.table import Table
from mimic.collector import collect_movements
from mimic.lstm_trainer import train_lstm
from mimic.spoofer import spoof_and_plot
from mimic.visuals import display_intro, display_status, display_success, display_error

console = Console()

CSV_PATH = "data/mouse_data.csv"
MODEL_PATH = "models/mimic_lstm.pt"
SCALER_PATH = "models/mimic_scaler.pkl"

def show_menu():
    table = Table(title="🎮 M.I.M.I.C. Control Panel")
    table.add_column("Option", style="cyan", no_wrap=True)
    table.add_column("Action", style="magenta")

    table.add_row("1", "🖱️ Collect mouse data")
    table.add_row("2", "🧠 Train LSTM model")
    table.add_row("3", "🤖 Run cursor spoofer (Live + Graph)")
    table.add_row("4", "❌ Exit")

    console.print(table)

def main():
    display_intro()
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    while True:
        show_menu()
        choice = input("Enter choice: ").strip()

        if choice == "1":
            secs = input("How many seconds to record? (default 30): ").strip()
            duration = int(secs) if secs.isdigit() else 30
            display_status(f"Collecting mouse data for {duration} seconds")
            collect_movements(CSV_PATH, duration=duration)
            display_success(f"Data saved to {CSV_PATH}")

        elif choice == "2":
            if not os.path.exists(CSV_PATH):
                display_error("No data file found. Please record mouse data first (option 1).")
                continue

            if os.path.exists(MODEL_PATH):
                overwrite = input("⚠️ A trained model already exists. Retrain? (y/n): ").strip().lower()
                if overwrite != "y":
                    display_status("Training cancelled. Using existing model.")
                    continue

            display_status(f"Training LSTM model on {CSV_PATH}")
            try:
                train_lstm(CSV_PATH, model_path=MODEL_PATH, scaler_path=SCALER_PATH)
                display_success("Training complete!")
            except Exception as e:
                display_error(f"Error during training: {e}")

        elif choice == "3":
            if not os.path.exists(CSV_PATH):
                display_error("No CSV data found. Please record mouse data first (option 1).")
                continue
            if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
                display_error("No trained model found. Please train one first (option 2).")
                continue

            display_status("Running cursor spoofer... Graph will pop up automatically!")
            try:
                spoof_and_plot(model_path=MODEL_PATH, scaler_path=SCALER_PATH, csv_path=CSV_PATH)
                display_success("Spoofing finished.")
            except Exception as e:
                display_error(f"Error during spoofing: {e}")

        elif choice == "4":
            display_status("Exiting... Goodbye!")
            sys.exit(0)

        else:
            display_error("Invalid choice, try again.")

if __name__ == "__main__":
    main()
