import numpy as np
import pandas as pd

from modelo_credito.eda import iv_ranking, summarize_variable, woe_iv_table
from modelo_credito.features import create_features
from modelo_credito.selection import (
    calculate_vif,
    filter_by_iv,
    remove_high_correlation,
    select_variables,
)


def test_woe_iv_table_returns_finite_values(sample_df):
    tabela, iv = woe_iv_table(sample_df, "PAY_0", "default", is_continuous=False)

    assert not tabela.empty
    assert np.isfinite(iv)
    assert {"bin", "total", "bads", "goods", "bad_rate", "woe", "iv_bin"} <= set(tabela.columns)


def test_iv_ranking_and_summary(sample_df):
    features = create_features(sample_df)
    ranking = iv_ranking(
        features,
        ["PAY_0", "UTIL_MEDIA", "AGE"],
        "default",
        continuous_features={"UTIL_MEDIA", "AGE"},
    )

    assert ranking["IV"].is_monotonic_decreasing

    resumo, woe = summarize_variable(
        features,
        "UTIL_MEDIA",
        "default",
        is_continuous=True,
    )

    assert resumo["Variável"] == "UTIL_MEDIA"
    assert np.isfinite(resumo["IV"])
    assert len(woe) == len(features)


def test_selection_filters_and_reduces_redundancy(sample_df):
    features = create_features(sample_df)
    features["COPIA_UTIL"] = features["UTIL_MEDIA"]

    iv = pd.Series(
        {
            "PAY_0": 0.80,
            "UTIL_MEDIA": 0.20,
            "COPIA_UTIL": 0.15,
            "AGE": 0.01,
        }
    )

    filtradas = filter_by_iv(list(iv.index), iv, threshold=0.02)
    assert "AGE" not in filtradas

    selecionadas = remove_high_correlation(
        features,
        ["UTIL_MEDIA", "COPIA_UTIL"],
        iv,
        threshold=0.70,
    )
    assert selecionadas == ["UTIL_MEDIA"]

    vif = calculate_vif(features, ["PAY_0", "UTIL_MEDIA"])
    assert set(vif.index) == {"PAY_0", "UTIL_MEDIA"}

    final = select_variables(
        features,
        ["PAY_0", "UTIL_MEDIA", "COPIA_UTIL", "AGE"],
        iv,
    )
    assert "AGE" not in final
    assert "COPIA_UTIL" not in final
