from pathlib import Path


def load_data(path: Path) -> None:
    if path.exists():
        print("pth exits")
    else:
        print(f"Data path {path} does not exist. Please run the data loading script.")
        return None
