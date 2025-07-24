from mimic import collector, model, spoofer, visuals
import sys

def main():
    visuals.intro_sequence()

    while True:
        print("\n[1] Collect mouse data")
        print("[2] Train model")
        print("[3] Run cursor spoofer")
        print("[4] Exit\n")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            duration = input("Duration in seconds to collect? [default 15]: ").strip()
            try:
                duration = int(duration)
            except:
                duration = 15
            collector.collect_movements(duration=duration)

        elif choice == "2":
            csv_path = "data/mouse_movements.csv"
            mdl = model.train_model(csv_path)
            if mdl:
                model.save_model(mdl)

        elif choice == "3":
            mdl = model.load_model()
            spoofer.simulate_movement(mdl)

        elif choice == "4":
            print("Exiting.")
            sys.exit()

        else:
            print("Invalid option.")

if __name__ == "__main__":
    main()
