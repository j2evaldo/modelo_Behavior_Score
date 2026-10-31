from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import GREEN_ACCENT, GREEN_DARK, GREEN_LIGHT, GREEN_MID, GREEN_PALE
from .eda import iv_ranking


def _finish(fig, save_path: str | Path | None = None, show: bool = True) -> None:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_estabilidade_safra(
    df: pd.DataFrame,
    col_safra: str,
    target: str,
    show: bool = True,
    save_path: str | Path | None = None,
):
    safra_summary = (
        df.groupby(col_safra, observed=False)
        .agg(volume=(target, "count"), bad_rate=(target, "mean"))
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    ax1.bar(
        safra_summary[col_safra],
        safra_summary["volume"],
        color=GREEN_PALE,
        edgecolor=GREEN_MID,
        alpha=0.7,
        label="Volume Ativo",
    )
    ax2.plot(
        safra_summary[col_safra],
        safra_summary["bad_rate"] * 100,
        color=GREEN_DARK,
        marker="o",
        linewidth=2.5,
        label="Bad Rate (%)",
    )

    ax1.set_xlabel("Safra de Referência", fontweight="bold", color=GREEN_DARK)
    ax1.set_ylabel("Volume de Clientes Ativos", color=GREEN_MID, fontweight="bold")
    ax2.set_ylabel("Taxa de Default (%)", color=GREEN_DARK, fontweight="bold")
    ax1.grid(False)
    ax2.grid(True, linestyle=":", alpha=0.6)
    plt.xticks(rotation=45)
    plt.title(
        "Estabilidade de Risco no Tempo: Volume vs. Taxa de Default",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    plt.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax1, ax2


def plot_ranking_iv(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    bins: int = 10,
    continuous_features: set[str] | None = None,
    show: bool = True,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    iv_df = iv_ranking(
        df,
        features,
        target,
        bins=bins,
        continuous_features=continuous_features,
    ).sort_values("IV", ascending=True)

    cores = []
    for val in iv_df["IV"]:
        if val < 0.02:
            cores.append("#D4EFDF")
        elif val < 0.10:
            cores.append("#A9DFBF")
        elif val < 0.30:
            cores.append("#52BE80")
        elif val <= 0.50:
            cores.append("#1E8449")
        else:
            cores.append("#114B27")

    fig, ax = plt.subplots(figsize=(10, max(5, len(features) * 0.45)))
    ax.barh(iv_df["Feature"], iv_df["IV"], color=cores, edgecolor=GREEN_DARK, linewidth=0.6)

    ax.axvline(0.02, color=GREEN_LIGHT, linestyle=":", label="Mínimo Relevante (0.02)")
    ax.axvline(0.10, color=GREEN_MID, linestyle="--", label="Preditores Médios (0.10)")
    ax.axvline(0.30, color=GREEN_DARK, linestyle="--", label="Preditores Fortes (0.30)")
    ax.axvline(0.50, color=GREEN_ACCENT, linestyle="-", label="Alerta de Vazamento (>0.50)")

    ax.set_xlabel("Information Value (IV)", fontweight="bold", color=GREEN_DARK)
    ax.set_title(
        "Screening Univariado: Information Value (IV)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return iv_df.sort_values("IV", ascending=False).reset_index(drop=True)


def plot_curva_monotonicidade(
    tabela_woe: pd.DataFrame,
    feature_name: str,
    total_iv: float,
    show: bool = True,
    save_path: str | Path | None = None,
):
    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(tabela_woe))

    ax1.bar(
        x,
        tabela_woe["bad_rate"] * 100,
        color=GREEN_MID,
        alpha=0.65,
        width=0.45,
        label="Bad Rate (%)",
    )
    ax2.plot(
        x,
        tabela_woe["woe"],
        color=GREEN_DARK,
        marker="s",
        markersize=6,
        linewidth=2.5,
        label="WoE [ln(Good%/Bad%)]",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(tabela_woe["bin"], rotation=35, ha="right")
    ax1.set_ylabel("Taxa de Default (% Bad)", color=GREEN_MID, fontweight="bold")
    ax2.set_ylabel("Weight of Evidence (WoE)", color=GREEN_DARK, fontweight="bold")
    ax1.set_title(
        f"Monotonicidade e Risco: {feature_name} (IV: {total_iv:.3f})",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    ax1.grid(False)
    ax2.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax1, ax2


def plot_matriz_spearman(
    df: pd.DataFrame,
    top_features: list[str],
    show: bool = True,
    save_path: str | Path | None = None,
):
    from scipy.stats import spearmanr

    corr, _ = spearmanr(df[top_features].dropna())
    corr_df = pd.DataFrame(corr, index=top_features, columns=top_features)

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap_verdes = sns.light_palette(GREEN_DARK, as_cmap=True)

    sns.heatmap(
        corr_df,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap=cmap_verdes,
        vmin=0,
        vmax=1,
        linewidths=0.6,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(
        "Matriz de Correlação de Spearman (Verificação de Redundância)",
        fontsize=12,
        fontweight="bold",
        pad=15,
    )
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax


def plot_gains(
    tabela: pd.DataFrame,
    title: str = "Curva de ganhos acumulados",
    show: bool = True,
    save_path: str | Path | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [0] + tabela["pct_pop_acum"].tolist(),
        [0] + tabela["pct_bads_acum"].tolist(),
        color=GREEN_DARK,
        marker="o",
        linewidth=2.5,
        label="Modelo",
    )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Aleatório")

    for fracao, rotulo in [(0.10, "10%"), (0.20, "20%"), (0.30, "30%")]:
        idx = (tabela["pct_pop_acum"] >= fracao).idxmax()
        x = tabela.loc[idx, "pct_pop_acum"]
        y = tabela.loc[idx, "pct_bads_acum"]
        ax.scatter(x, y, color=GREEN_MID, zorder=5)
        ax.annotate(f"{rotulo}: {y:.1%}", (x, y), textcoords="offset points", xytext=(6, -12))

    ax.set_xlabel("% da carteira priorizada (maior risco primeiro)")
    ax.set_ylabel("% de maus pagadores capturados")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax


def plot_lift(
    tabela: pd.DataFrame,
    title: str = "Lift por decil",
    show: bool = True,
    save_path: str | Path | None = None,
):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.bar(
        tabela["decil"],
        tabela["lift"],
        color=GREEN_PALE,
        edgecolor=GREEN_MID,
        alpha=0.8,
        label="Lift por decil",
    )
    ax1.axhline(1, color="gray", linestyle="--", linewidth=1)
    ax1.set_xlabel("Decil de risco (1 = maior risco)")
    ax1.set_ylabel("Lift por decil", color=GREEN_MID)
    ax1.set_xticks(tabela["decil"])

    ax2.plot(
        tabela["decil"],
        tabela["lift_acum"],
        color=GREEN_DARK,
        marker="o",
        linewidth=2.5,
        label="Lift acumulado",
    )
    ax2.set_ylabel("Lift acumulado", color=GREEN_DARK)

    ax1.set_title(title)
    ax1.grid(False)
    ax2.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax1, ax2


def plot_calibration(
    y: np.ndarray,
    probabilities: np.ndarray,
    title: str = "Curva de calibração",
    n_bins: int = 10,
    show: bool = True,
    save_path: str | Path | None = None,
):
    from sklearn.calibration import calibration_curve

    prob_true, prob_pred = calibration_curve(
        y,
        probabilities,
        n_bins=n_bins,
        strategy="quantile",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, color=GREEN_DARK, label="Modelo")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfeito")
    ax.set_xlabel("Probabilidade prevista")
    ax.set_ylabel("Bad rate observado")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    _finish(fig, save_path=save_path, show=show)
    return fig, ax
