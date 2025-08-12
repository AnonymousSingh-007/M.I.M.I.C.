# import sys
# from rich.console import Console
# from mimic import spoofer, model, visuals

# console = Console()


# def main():
#     visuals.intro_sequence()

#     while True:
#         console.print("\n[bold blue]Main Menu:[/bold blue]")
#         console.print("[cyan][1][/cyan] 🖱️ Collect mouse data")
#         console.print("[cyan][2][/cyan] 🧠 Train model")
#         console.print("[cyan][3][/cyan] 🤖 Run cursor spoofer")
#         console.print("[cyan][4][/cyan] ❌ Exit\n")

#         choice = input("Choose an option: ").strip()

#         if choice == "1":
#             duration = input("Duration in seconds to collect? [default 15]: ").strip()
#             try:
#                 duration = int(duration)
#             except:
#                 duration = 15
#             spoofer.record_mouse_movements(duration=duration)

#         elif choice == "2":
#             csv_path = input("Path to CSV file (default: data/mouse_movements.csv): ").strip()
#             if not csv_path:
#                 csv_path = "data/mouse_movements.csv"
#             model.train(csv_path)

#         elif choice == "3":
#             mdl = model.load_model()
#             spoofer.simulate_movement(mdl)

#         elif choice == "4":
#             console.print("[bold red]👋 Exiting M.I.M.I.C.[/bold red]")
#             sys.exit()

#         else:
#             console.print("[bold red]Invalid option. Please choose 1–4.[/bold red]")


# if __name__ == "__main__":
#     main()

#trying dynamic time
import sys
from rich.console import Console
from mimic import spoofer, model, visuals

console = Console()
recorded_duration = 0  # Tracks how long data was collected


def main():
    global recorded_duration
    visuals.intro_sequence()

    while True:
        console.print("\n[bold blue]📋 Main Menu:[/bold blue]")
        console.print("[cyan][1][/cyan] 🖱️ Collect mouse data")
        console.print("[cyan][2][/cyan] 🧠 Train model")
        console.print("[cyan][3][/cyan] 🤖 Run cursor spoofer")
        console.print("[cyan][4][/cyan] ❌ Exit\n")

        choice = input("Choose an option: ").strip()

        # 1️⃣ Data collection
        if choice == "1":
            duration = input("Duration in seconds to collect? [default 15]: ").strip()
            try:
                duration = int(duration)
            except ValueError:
                duration = 15

            recorded_duration = duration
            spoofer.record_mouse_movements(duration=duration)

        # 2️⃣ Training
        elif choice == "2":
            csv_path = input("Path to CSV file (default: data/mouse_movements.csv): ").strip()
            if not csv_path:
                csv_path = "data/mouse_movements.csv"

            trained_model = model.train(csv_path)
            if trained_model:
                console.print(f"[green]✅ Model trained successfully from {csv_path}[/green]")

        # 3️⃣ Spoofing
        elif choice == "3":
            if recorded_duration == 0:
                console.print("[bold red]⚠️ Please collect mouse data first![/bold red]")
                continue

            max_spoof = int(0.75 * recorded_duration)
            default_spoof = min(5, max_spoof)
            prompt = f"Enter spoofing duration (max {max_spoof}s) [default {default_spoof}]: "
            spoof_duration = input(prompt).strip()

            try:
                spoof_duration = int(spoof_duration)
                if spoof_duration > max_spoof:
                    console.print(f"[yellow]⚠️ Duration capped to {max_spoof}s[/yellow]")
                    spoof_duration = max_spoof
            except ValueError:
                spoof_duration = default_spoof

            try:
                mdl = model.load_model()
                spoofer.simulate_movement(mdl, duration=spoof_duration)
            except FileNotFoundError:
                console.print("[bold red]❌ No trained model found! Please train the model first.[/bold red]")

        # 4️⃣ Exit
        elif choice == "4":
            console.print("[bold red]👋 Exiting M.I.M.I.C.[/bold red]")
            sys.exit()

        else:
            console.print("[bold red]❌ Invalid option. Please choose 1–4.[/bold red]")


if __name__ == "__main__":
    main()
