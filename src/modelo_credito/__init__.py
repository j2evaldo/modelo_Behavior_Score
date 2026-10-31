from . import calibration, data, eda, evaluation, features, modeling, selection
from .config import TARGET

__all__ = [
    "TARGET",
    "calibration",
    "data",
    "eda",
    "evaluation",
    "features",
    "modeling",
    "selection",
]


def main() -> None:
    print("Hello from modelo-credito!")
