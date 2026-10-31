import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def filter_by_iv(
    features: list[str],
    iv: pd.Series | dict[str, float],
    threshold: float = 0.02,
) -> list[str]:
    return [feature for feature in features if iv.get(feature, 0) >= threshold]


def remove_high_correlation(
    df: pd.DataFrame,
    features: list[str],
    iv: pd.Series | dict[str, float],
    threshold: float = 0.70,
) -> list[str]:
    selecionadas = features.copy()

    while len(selecionadas) > 1:
        corr = df[selecionadas].corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pares = upper.stack().sort_values(ascending=False)

        if pares.empty or pares.iloc[0] <= threshold:
            break

        var_a, var_b = pares.index[0]
        remover = var_a if iv.get(var_a, 0) < iv.get(var_b, 0) else var_b
        selecionadas.remove(remover)

    return selecionadas


def calculate_vif(df: pd.DataFrame, features: list[str]) -> pd.Series:
    X = df[features]
    X_centrado = X - X.mean()
    valores = [
        variance_inflation_factor(X_centrado.values, i)
        for i in range(X.shape[1])
    ]
    return pd.Series(valores, index=features)


def select_by_vif(
    df: pd.DataFrame,
    features: list[str],
    threshold: float = 5.0,
) -> list[str]:
    selecionadas = features.copy()

    while len(selecionadas) > 1:
        vif = calculate_vif(df, selecionadas)
        if vif.max() <= threshold:
            break
        selecionadas.remove(vif.idxmax())

    return selecionadas


def select_variables(
    df: pd.DataFrame,
    features: list[str],
    iv: pd.Series | dict[str, float],
    iv_threshold: float = 0.02,
    correlation_threshold: float = 0.70,
    vif_threshold: float = 5.0,
) -> list[str]:
    selecionadas = filter_by_iv(features, iv, threshold=iv_threshold)
    selecionadas = remove_high_correlation(
        df,
        selecionadas,
        iv,
        threshold=correlation_threshold,
    )
    selecionadas = select_by_vif(df, selecionadas, threshold=vif_threshold)
    return selecionadas
