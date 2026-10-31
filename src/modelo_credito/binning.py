import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def fit_binning_process(
    train: pd.DataFrame,
    variable_names: list[str],
    categorical_variables: list[str] | None = None,
    target: str = "default",
    min_prebin_size: float = 0.05,
    max_n_bins: int = 5,
    **kwargs: Any,
):
    from optbinning import BinningProcess

    process = BinningProcess(
        variable_names=variable_names,
        categorical_variables=categorical_variables,
        min_prebin_size=min_prebin_size,
        max_n_bins=max_n_bins,
        **kwargs,
    )
    process.fit(train[variable_names], train[target])
    return process


def transform(
    process,
    base: pd.DataFrame,
    variable_names: list[str],
    metric: str = "woe",
) -> pd.DataFrame:
    return process.transform(base[variable_names], metric=metric)


def build_transformed_dataset(
    process,
    base: pd.DataFrame,
    variable_names: list[str],
    metadata: list[str],
    target: str,
    metric: str = "woe",
) -> pd.DataFrame:
    transformed = transform(process, base, variable_names, metric=metric)
    return pd.concat(
        [
            base[metadata + [target]].reset_index(drop=True),
            transformed.reset_index(drop=True),
        ],
        axis=1,
    )


def binning_summary(process) -> pd.DataFrame:
    return process.summary().sort_values("iv", ascending=False).reset_index(drop=True)


def save_binning_process(process, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(process, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_binning_process(path: str | Path):
    with open(path, "rb") as file:
        return pickle.load(file)
