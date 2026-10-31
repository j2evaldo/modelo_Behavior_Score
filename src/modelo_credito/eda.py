import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def woe_iv_table(
    df: pd.DataFrame,
    feature: str,
    target: str,
    bins: int = 10,
    is_continuous: bool = True,
) -> tuple[pd.DataFrame, float]:
    tmp = df[[feature, target]].copy()
    tmp["bin"] = tmp[feature].fillna("MISSING").astype(object)

    if is_continuous:
        mask_valid = tmp[feature].notnull()
        try:
            tmp.loc[mask_valid, "bin"] = pd.qcut(
                tmp.loc[mask_valid, feature],
                q=bins,
                duplicates="drop",
            ).astype(str)
        except ValueError:
            tmp.loc[mask_valid, "bin"] = pd.cut(
                tmp.loc[mask_valid, feature],
                bins=bins,
                duplicates="drop",
            ).astype(str)

    total_goods = (tmp[target] == 0).sum()
    total_bads = (tmp[target] == 1).sum()

    agg = (
        tmp.groupby("bin", observed=False)
        .agg(
            total=(target, "count"),
            bads=(target, "sum"),
            goods=(target, lambda x: (x == 0).sum()),
            bad_rate=(target, "mean"),
        )
        .reset_index()
    )

    agg["dist_good"] = (agg["goods"] + 0.5) / (total_goods + 1.0)
    agg["dist_bad"] = (agg["bads"] + 0.5) / (total_bads + 1.0)
    agg["woe"] = np.log(agg["dist_good"] / agg["dist_bad"])
    agg["iv_bin"] = (agg["dist_good"] - agg["dist_bad"]) * agg["woe"]

    return agg, float(agg["iv_bin"].sum())


def iv_ranking(
    df: pd.DataFrame,
    features: list[str],
    target: str,
    bins: int = 10,
    continuous_features: set[str] | None = None,
) -> pd.DataFrame:
    continuous_features = set(features) if continuous_features is None else set(continuous_features)

    rows = []
    for feature in features:
        _, iv = woe_iv_table(
            df,
            feature,
            target,
            bins=bins,
            is_continuous=feature in continuous_features,
        )
        rows.append({"Feature": feature, "IV": iv})

    return pd.DataFrame(rows).sort_values("IV", ascending=False).reset_index(drop=True)


def monotonicity_label(bad_rate: np.ndarray, rho: float) -> str:
    crescente = len(bad_rate) > 1 and np.all(np.diff(bad_rate) >= 0)
    decrescente = len(bad_rate) > 1 and np.all(np.diff(bad_rate) <= 0)

    if crescente or decrescente:
        return "Sim (crescente)" if bad_rate[-1] >= bad_rate[0] else "Sim (decrescente)"
    if abs(rho) >= 0.90:
        return "Sim (crescente)" if rho > 0 else "Sim (decrescente)"
    if abs(rho) >= 0.70:
        return "Parcial"
    return "Não"


def summarize_variable(
    df: pd.DataFrame,
    feature: str,
    target: str,
    is_continuous: bool,
    bins: int = 10,
) -> tuple[dict[str, object], pd.Series]:
    tmp = df[[feature, target]].copy()
    tmp["faixa"] = "MISSING"

    if is_continuous:
        mask_valid = tmp[feature].notnull()
        faixa_valid = pd.qcut(
            tmp.loc[mask_valid, feature],
            q=bins,
            duplicates="drop",
        )
        tmp.loc[mask_valid, "faixa"] = faixa_valid.astype(str)
        categorias = [str(c) for c in faixa_valid.cat.categories]
        if (~mask_valid).any():
            categorias.append("MISSING")
    elif feature in ("PAY_0", "PAY_2"):
        agrupada = tmp[feature].clip(lower=0, upper=3)
        tmp["faixa"] = agrupada.astype(int).astype(str)
        categorias = [str(int(v)) for v in sorted(agrupada.dropna().unique())]
    elif feature == "FREQ_ATRASO":
        agrupada = tmp[feature].clip(upper=3)
        tmp["faixa"] = agrupada.astype(int).astype(str)
        categorias = [str(int(v)) for v in sorted(agrupada.dropna().unique())]
    else:
        tmp["faixa"] = tmp[feature].astype(str)
        categorias = sorted(tmp["faixa"].dropna().unique())

    tmp["faixa"] = pd.Categorical(tmp["faixa"], categories=categorias, ordered=True)

    total_goods = (tmp[target] == 0).sum()
    total_bads = (tmp[target] == 1).sum()

    agg = (
        tmp.groupby("faixa", observed=False)
        .agg(
            total=(target, "count"),
            bads=(target, "sum"),
            goods=(target, lambda x: (x == 0).sum()),
            bad_rate=(target, "mean"),
        )
        .reset_index()
    )

    agg["dist_good"] = (agg["goods"] + 0.5) / (total_goods + 1.0)
    agg["dist_bad"] = (agg["bads"] + 0.5) / (total_bads + 1.0)
    agg["woe"] = np.log(agg["dist_good"] / agg["dist_bad"])
    agg["iv_bin"] = (agg["dist_good"] - agg["dist_bad"]) * agg["woe"]

    validos = agg[(agg["total"] > 0) & (agg["faixa"].astype(str) != "MISSING")]
    bad_rate = validos["bad_rate"].values
    rho = (
        spearmanr(np.arange(len(bad_rate)), bad_rate).statistic
        if len(bad_rate) > 2
        else np.nan
    )

    resumo = {
        "Variável": feature,
        "IV": agg["iv_bin"].sum(),
        "Monotônica": monotonicity_label(bad_rate, rho),
        "ρ (Spearman)": rho,
        "WoE mín": agg["woe"].min(),
        "WoE máx": agg["woe"].max(),
        "Amplitude WoE": agg["woe"].max() - agg["woe"].min(),
    }

    if feature in ("PAY_0", "PAY_2"):
        chave = df[feature].clip(lower=0, upper=3).astype(int).astype(str)
    elif feature == "FREQ_ATRASO":
        chave = df[feature].clip(upper=3).astype(int).astype(str)
    elif is_continuous:
        chave = pd.Series("MISSING", index=df.index)
        chave.loc[mask_valid] = faixa_valid.astype(str)
    else:
        chave = df[feature].astype(str)

    mapa_woe = agg.set_index(agg["faixa"].astype(str))["woe"]
    woe_serie = chave.map(mapa_woe).astype(float)

    return resumo, woe_serie
