from modelo_credito.binning import (
    binning_summary,
    fit_binning_process,
    transform,
)
from modelo_credito.features import create_features


def test_binning_process_fit_and_transform(sample_df):
    base = create_features(sample_df)
    variaveis = ["PAY_0", "LIMIT_BAL", "UTIL_MEDIA"]

    process = fit_binning_process(
        base,
        variaveis,
        categorical_variables=["PAY_0"],
        max_n_bins=3,
    )
    woe = transform(process, base, variaveis, metric="woe")
    resumo = binning_summary(process)

    assert woe.shape == (len(base), len(variaveis))
    assert set(resumo["name"]) == set(variaveis)
    assert resumo["iv"].notna().all()
