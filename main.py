import sys
import time
from pathlib import Path

from mimic import visuals
from mimic.collector import collect_movements

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

LSTM_MODEL_PATH = MODELS_DIR / "lstm_fixed.pt"
LSTM_SCALER_PATH = MODELS_DIR / "scaler_fixed.pkl"


def show_menu():
    visuals.display_intro()
    print("[1] Collect mouse data")
    print("[2] Train LSTM on CSV")
    print("[3] Run LSTM spoofer")
    print("[9] Exit\n")


def main():
    while True:
        try:
            show_menu()
            choice = input("Choice → ").strip()

            if choice == "1":
                secs = input("Duration seconds (default 120): ").strip()
                duration = int(secs) if secs.isdigit() else 120
                fname = DATA_DIR / f"session_{int(time.time())}.csv"
                visuals.display_status("Collecting mouse data")
                collect_movements(str(fname), duration)
                visuals.display_success("Data collected")

            elif choice == "2":
                from mimic.lstm_trainer import train_lstm_fixed

                csv_name = input("CSV filename (inside data/): ").strip()
                csv_path = DATA_DIR / csv_name

                if not csv_path.exists():
                    visuals.display_error("CSV not found")
                    continue

                visuals.display_status("Training LSTM")
                train_lstm_fixed(
                    csv_file=str(csv_path),
                    model_path=str(LSTM_MODEL_PATH),
                    scaler_path=str(LSTM_SCALER_PATH)
                )
                visuals.display_success("Training complete")

            elif choice == "3":
                from mimic.spoofer_fixed import spoof_realistic
                spoof_realistic(
                    str(LSTM_MODEL_PATH),
                    str(LSTM_SCALER_PATH)
                )

            elif choice == "9":
                sys.exit(0)

            else:
                visuals.display_error("Invalid choice")

        except KeyboardInterrupt:
            sys.exit(0)


if __name__ == "__main__":
    main()
