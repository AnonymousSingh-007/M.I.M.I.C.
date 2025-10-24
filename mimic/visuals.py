# #mimic/visuals.py
# from rich.console import Console
# from rich.panel import Panel
# from rich.text import Text
# from rich.box import ROUNDED
# from rich.columns import Columns

# console = Console()

# __all__ = [
#     "display_intro",
#     "display_status",
#     "display_success",
#     "display_error",
# ]


# def display_intro():
#     """Stylized welcome panel for M.I.M.I.C."""
#     title = Text("M.I.M.I.C.", style="bold #00ffff")
#     subtitle = Text("Motion Imitation Mechanism for Input Camouflage", style="#ff00ff")
#     tagline = Text("“Looks real. Isn’t.”", style="bold green", justify="center")

#     panel_content = Columns(
#         [
#             Text.from_markup("[bold #ff8800]M[/]achine [#ff8800]I[/]ntelligence\n"
#                              "[bold #ff8800]M[/]imicking [#ff8800]I[/]nput [#ff8800]C[/]amouflage",
#                              justify="center"),
#             Text.from_markup("[bold green]Welcome to M.I.M.I.C, a machine learning-driven\n"
#                              "cursor spoofer that emulates human mouse\n"
#                              "movements with uncanny precision.[/bold green]",
#                              justify="center")
#         ],
#         padding=(1, 2)
#     )

#     console.print(Panel(panel_content, title=title, subtitle=subtitle,
#                         border_style="bold #888888", box=ROUNDED))
#     console.print(tagline, justify="center")
#     console.print("")


# def display_status(message: str) -> None:
#     """Show a yellow status message."""
#     console.print(f"[bold yellow]→[/bold yellow] [yellow]{message}...[/yellow]")


# def display_success(message: str) -> None:
#     """Show a green success message."""
#     console.print(f"[bold green]✔[/bold green] [green]{message}[/green]")


# def display_error(message: str) -> None:
#     """Show a red error message."""
#     console.print(f"[bold red]❌[/bold red] [red]{message}[/red]")


#mimic/visuals.py


# mimic/visuals.py
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.box import ROUNDED, DOUBLE
from rich.columns import Columns
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.align import Align
from rich.markdown import Markdown
import time

console = Console()

# =========================
# Intro / Splash
# =========================
def display_intro() -> None:
    title = Text("🎮 M.I.M.I.C.", style="bold cyan underline")
    subtitle = Text("Motion Imitation Mechanism for Input Camouflage", style="magenta")
    
    content = Columns([
        Text(
            "[bold orange]M[/]achine [orange]I[/]ntelligence\n[bold orange]M[/]imicking [orange]I[/]nput [orange]C[/]amouflage", 
            justify="center",
            style="bold yellow"
        ),
        Text(
            "[green]M.I.M.I.C: a cursor spoofer emulating human-like movement.[/green]\n:joystick: Ready to play!", 
            justify="center"
        )
    ], padding=(1, 2))
    
    console.print(Panel(content, title=title, subtitle=subtitle, border_style="bright_blue", box=DOUBLE))
    console.print(Text("“Looks real. Isn’t.”", style="bold green"), justify="center")
    console.print(Rule(style="grey50"))

# =========================
# Status / Gamification
# =========================
def display_status(msg: str) -> None:
    console.print(Panel(f"[yellow]→ {msg}[/yellow]", title="Status", border_style="yellow", box=ROUNDED))

def display_success(msg: str) -> None:
    console.print(Panel(f"[green]✔ {msg}[/green]", title="Success", border_style="green", box=ROUNDED))

def display_error(msg: str) -> None:
    console.print(Panel(f"[red]❌ {msg}[/red]", title="Error", border_style="red", box=ROUNDED))

# =========================
# Progress / Gamified Bars
# =========================
def show_progress_bar(task_name: str, total: int = 100, speed: float = 0.02):
    console.print(f"\n[bold magenta]Starting: {task_name}[/bold magenta]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None, complete_style="green", finished_style="bright_green"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(task_name, total=total)
        for i in range(total):
            time.sleep(speed)
            progress.update(task, advance=1)
        progress.refresh()
    console.print(f"[bold green]✔ {task_name} complete![/bold green]\n")

# =========================
# Fun / Gamification Panels
# =========================
def show_level_panel(level: int, xp: int, goal: int):
    percentage = min(int((xp/goal)*100), 100)
    bar = f"[green]{'■' * (percentage // 5)}[/green][grey37]{'□' * (20 - (percentage // 5))}[/grey37]"
    panel_content = f"Level: [bold cyan]{level}[/bold cyan]\nXP: {xp}/{goal}\n{bar}"
    console.print(Panel(panel_content, title="Player Progress", border_style="bright_cyan", box=ROUNDED))

def show_achievement(name: str, desc: str):
    console.print(Panel(f":sparkles: [bold yellow]{name}[/bold yellow]\n[grey70]{desc}[/grey70]", 
                        border_style="gold1", box=ROUNDED, title="Achievement Unlocked!"))

# =========================
# Fancy Divider
# =========================
def show_divider(msg: str = ""):
    if msg:
        console.print(Rule(title=msg, style="bright_magenta"))
    else:
        console.print(Rule(style="bright_magenta"))
