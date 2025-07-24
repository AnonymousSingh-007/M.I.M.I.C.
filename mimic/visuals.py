from rich.console import Console
from rich.panel import Panel

console = Console()

def intro_sequence():
    console.print(Panel.fit(
        "[bold cyan]M.I.M.I.C.[/bold cyan]\n"
        "[magenta]Motion Imitation Mechanism for Input Camouflage[/magenta]\n"
        "[green]“Looks real. Isn’t.”[/green]",
        title="💠 Welcome 💠"
    ))
