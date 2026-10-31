import numpy as np

from modelo_credito.config import BILL_COLUMNS, PAY_AMT_COLUMNS, PAY_COLUMNS
from modelo_credito.data import (
    apply_treatment,
    compute_treatment_limits,
    temporal_split,
)
from modelo_credito.features import create_features


def test_temporal_split_preserves_temporal_order(sample_df):
    treino, validacao, teste = temporal_split(sample_df)

    assert treino["SAFRA"].max() < 200604
    assert set(validacao["SAFRA"].unique()) <= {200604, 200605}
    assert set(teste["SAFRA"].unique()) <= {200606, 200607}
    assert len(treino) + len(validacao) + len(teste) == len(sample_df)


def test_treatment_uses_train_limits_and_does_not_mutate(sample_df):
    treino, _, _ = temporal_split(sample_df)
    limites = compute_treatment_limits(treino, PAY_COLUMNS + PAY_AMT_COLUMNS)
    original = treino.copy()

    tratado = apply_treatment(treino, limites)

    assert treino.equals(original)
    for col in PAY_COLUMNS + PAY_AMT_COLUMNS:
        assert tratado[col].max() <= limites[col]
    for col in BILL_COLUMNS:
        assert tratado[col].min() >= -1


def test_create_features_adds_expected_columns(sample_df):
    resultado = create_features(sample_df)

    esperadas = {
        "SUM_BILL",
        "AVG_BILL",
        "MAX_BILL",
        "SUM_PAY",
        "AVG_PAY",
        "MAX_PAY",
        "UTILIZACAO",
        "UTIL_MEDIA",
        "PCT_PAGAMENTO",
        "FREQ_ATRASO",
        "MAX_ATRASO",
        "LIMIT_PER_CAPITA",
    }
    assert esperadas <= set(resultado.columns)
    assert len(resultado) == len(sample_df)
    assert resultado["UTILIZACAO"].dropna().between(0, 2).all()
    assert resultado["PCT_PAGAMENTO"].dropna().between(0, 2).all()
    assert np.isfinite(resultado["MAX_ATRASO"]).all()
