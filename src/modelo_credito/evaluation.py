import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from .modeling import predict_logit


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y, probabilities)
    fpr, tpr, _ = roc_curve(y, probabilities)
    ks = float(max(tpr - fpr))
    return {
        "n": len(y),
        "bad_rate": float(np.mean(y)),
        "auc": float(auc),
        "ks": ks,
        "gini": float(2 * auc - 1),
    }


def evaluate_model(
    model,
    base: pd.DataFrame,
    target: str,
    name: str,
) -> tuple[dict[str, object], np.ndarray]:
    probabilities = predict_logit(model, base)
    resumo = classification_metrics(base[target].values, probabilities)
    resumo["base"] = name
    return resumo, probabilities


def decision_table(
    base: pd.DataFrame,
    probabilities: np.ndarray,
    target: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = base[[target]].copy()
    df["prob"] = probabilities
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)
    df["decil"] = pd.qcut(df.index, n_bins, labels=False) + 1

    total_bads = df[target].sum()

    tabela = (
        df.groupby("decil")
        .agg(
            n=(target, "size"),
            bads=(target, "sum"),
            bad_rate=(target, "mean"),
            prob_media=("prob", "mean"),
            prob_min=("prob", "min"),
            prob_max=("prob", "max"),
        )
        .reset_index()
    )

    tabela["pct_populacao"] = tabela["n"] / tabela["n"].sum()
    tabela["pct_bads"] = tabela["bads"] / total_bads
    tabela["bad_rate_geral"] = df[target].mean()
    tabela["lift"] = tabela["bad_rate"] / tabela["bad_rate_geral"]
    tabela["bads_acum"] = tabela["bads"].cumsum()
    tabela["n_acum"] = tabela["n"].cumsum()
    tabela["bad_rate_acum"] = tabela["bads_acum"] / tabela["n_acum"]
    tabela["pct_bads_acum"] = tabela["bads_acum"] / total_bads
    tabela["pct_pop_acum"] = tabela["n_acum"] / tabela["n"].sum()
    tabela["lift_acum"] = tabela["bad_rate_acum"] / tabela["bad_rate_geral"]

    return tabela


def capture_summary(
    tabela: pd.DataFrame,
    fractions: tuple[float, ...] = (0.10, 0.20, 0.30, 0.50),
) -> pd.DataFrame:
    linhas = []

    for fracao in fractions:
        idx = (tabela["pct_pop_acum"] >= fracao).idxmax()
        linha = tabela.loc[idx]
        linhas.append(
            {
                "pct_carteira": fracao,
                "pct_maus_capturados": linha["pct_bads_acum"],
                "bad_rate_acumulado": linha["bad_rate_acum"],
                "lift_acumulado": linha["lift_acum"],
            }
        )

    return pd.DataFrame(linhas)


def cutoff_from_ks(y: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, probabilities)
    return float(thresholds[np.argmax(tpr - fpr)])


def confusion_at_cutoff(
    y: np.ndarray,
    probabilities: np.ndarray,
    cutoff: float,
) -> pd.DataFrame:
    predito = (probabilities >= cutoff).astype(int)
    matriz = confusion_matrix(y, predito)
    return pd.DataFrame(
        matriz,
        index=["Real 0", "Real 1"],
        columns=["Predito 0", "Predito 1"],
    )
