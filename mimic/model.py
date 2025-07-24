import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from rich.console import Console

console = Console()

def train_model(csv_path="data/mouse_movements.csv"):
    """
    Trains a linear regression model to predict (x, y) from timestamp.
    Returns trained model.
    """
    try:
        df = pd.read_csv(csv_path)

        if df.isnull().values.any():
            console.print("[red]❌ Data contains null values. Check CSV integrity.[/red]")
            return None

        X = df['timestamp'].to_numpy().reshape(-1, 1)
        y = df[['x', 'y']].values

        model = LinearRegression()
        model.fit(X, y)

        console.print(f"[green]✔ Model trained on {len(X)} data points[/green]")
        return model

    except FileNotFoundError:
        console.print(f"[red]❌ CSV not found at {csv_path}[/red]")
        return None


def save_model(model, path="model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    console.print(f"[cyan]💾 Model saved to {path}[/cyan]")


def load_model(path="model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    console.print(f"[green]✔ Model loaded from {path}[/green]")
    return model
