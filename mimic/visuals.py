#mimic/visuals.py
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED
from rich.columns import Columns

console = Console()

def display_intro():
    """Displays a stylized welcome panel for M.I.M.I.C."""
    title = Text("M.I.M.I.C.", style="bold #00ffff")
    subtitle = Text("Motion Imitation Mechanism for Input Camouflage", style="#ff00ff")
    
    tagline = Text("“Looks real. Isn’t.”", style="bold green", justify="center")

    panel_content = Columns(
        [
            Text.from_markup("[bold #ff8800]M[/][#ff8800]achine[/][#ff8800] I[/][#ff8800]ntelligence[/]\n"
                             "[bold #ff8800]M[/][#ff8800]imicking[/][bold #ff8800] I[/][#ff8800]nput[/][bold #ff8800] C[/][#ff8800]amouflage[/]",
                             justify="center"),
            Text.from_markup("[bold green]Welcome to M.I.M.I.C, a cutting-edge,\n"
                             "machine learning-driven cursor spoofer designed to\n"
                             "emulate human mouse movements with uncanny precision.[/bold green]",
                             justify="center")
        ],
        padding=(1, 2)
    )

    console.print(
        Panel(
            panel_content,
            title=title,
            subtitle=subtitle,
            border_style="bold #888888",
            box=ROUNDED
        )
    )
    console.print(tagline, justify="center")
    console.print("")

def display_status(message):
    """Displays a stylized status message."""
    console.print(f"[bold yellow]→[/bold yellow] [yellow]{message}...[/yellow]")

def display_success(message):
    """Displays a success message."""
    console.print(f"[bold green]✔[/bold green] [green]{message}[/green]")

def display_error(message):
    """Displays an error message."""
    console.print(f"[bold red]❌[/bold red] [red]{message}[/red]")
