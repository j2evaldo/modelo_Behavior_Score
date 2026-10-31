from pathlib import Path

import numpy as np
import pandas as pd

from .config import BILL_COLUMNS, PAY_AMT_COLUMNS, PAY_COLUMNS, TARGET


def load_dataset(path: str | Path, sep: str = ";") -> pd.DataFrame:
    return pd.read_csv(path, sep=sep)


def temporal_split(
    df: pd.DataFrame,
    safra_col: str = "SAFRA",
    train_end: int = 200604,
    validation_safras: tuple[int, ...] = (200604, 200605),
    test_safras: tuple[int, ...] = (200606, 200607),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    treino = df[df[safra_col] < train_end].copy()
    validacao = df[df[safra_col].isin(validation_safras)].copy()
    teste = df[df[safra_col].isin(test_safras)].copy()
    return treino, validacao, teste


def compute_treatment_limits(
    train: pd.DataFrame,
    columns: list[str] | None = None,
    quantile: float = 0.90,
) -> dict[str, float]:
    columns = columns or (PAY_COLUMNS + PAY_AMT_COLUMNS)
    return {col: float(train[col].quantile(quantile)) for col in columns}


def apply_treatment(
    base: pd.DataFrame,
    limits: dict[str, float],
    bill_columns: list[str] | None = None,
) -> pd.DataFrame:
    bill_columns = bill_columns or BILL_COLUMNS
    tratado = base.copy()

    for col in bill_columns:
        tratado[col] = np.where(tratado[col] < 0, -1, tratado[col])

    for col, limite in limits.items():
        tratado[col] = np.minimum(tratado[col], limite)

    return tratado


def save_splits(
    treino: pd.DataFrame,
    validacao: pd.DataFrame,
    teste: pd.DataFrame,
    output_dir: str | Path,
    suffix: str = "_tratado",
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    treino.to_parquet(output_dir / f"treino{suffix}.parquet")
    validacao.to_parquet(output_dir / f"validacao{suffix}.parquet")
    teste.to_parquet(output_dir / f"teste{suffix}.parquet")


def save_limits(limits: dict[str, float], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(limits).to_json(path)


def summary_by_safra(df: pd.DataFrame, safra_col: str = "SAFRA", target: str = TARGET) -> pd.DataFrame:
    return (
        df.groupby(safra_col)
        .agg(
            volume=(target, "count"),
            bads=(target, "sum"),
            bad_rate=(target, "mean"),
        )
        .reset_index()
    )
