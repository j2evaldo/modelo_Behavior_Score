import numpy as np
import pandas as pd

from modelo_credito.calibration import (
    apply_calibrator,
    choose_calibrator,
    fit_calibrators,
    logit,
)
from modelo_credito.evaluation import (
    capture_summary,
    classification_metrics,
    decision_table,
)
from modelo_credito.modeling import (
    build_formula,
    fit_logit_formula,
    predict_portable,
    save_portable_model,
    stepwise_logit,
)


def test_stepwise_removes_noise_variable(tmp_path):
    rng = np.random.default_rng(42)
    n = 2000
    x1 = rng.normal(size=n)
    ruido = rng.normal(size=n)
    prob = 1 / (1 + np.exp(-(-1 + 2 * x1)))
    y = rng.binomial(1, prob)

    df = pd.DataFrame({"x1": x1, "ruido": ruido, "default": y})
    selecionadas, modelo, historico = stepwise_logit(
        df,
        "default",
        ["x1", "ruido"],
        threshold=0.05,
    )

    assert selecionadas == ["x1"]
    assert "ruido" in historico["variavel_removida"].tolist()
    assert modelo.pvalues.max() <= 0.05


def test_portable_model_matches_statsmodels(tmp_path):
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    prob = 1 / (1 + np.exp(-(-1 + 1.5 * df["x1"] - 0.8 * df["x2"])))
    df["default"] = rng.binomial(1, prob)

    formula = build_formula("default", ["x1", "x2"])
    modelo = fit_logit_formula(df, formula)
    portable_path = tmp_path / "modelo_portavel.pkl"
    portable = save_portable_model(modelo, ["x1", "x2"], portable_path)

    esperado = modelo.predict(df).values
    obtido = predict_portable(df, portable)

    assert np.allclose(esperado, obtido)


def test_evaluation_metrics_and_decision_table():
    y = np.array([0] * 800 + [1] * 200)
    prob = np.linspace(0.01, 0.99, len(y))

    metricas = classification_metrics(y, prob)
    assert 0 <= metricas["auc"] <= 1
    assert 0 <= metricas["ks"] <= 1
    assert metricas["gini"] == 2 * metricas["auc"] - 1

    base = pd.DataFrame({"default": y})
    tabela = decision_table(base, prob, "default", n_bins=10)
    resumo = capture_summary(tabela)

    assert len(tabela) == 10
    assert tabela["lift_acum"].iloc[-1] == 1
    assert resumo["pct_maus_capturados"].is_monotonic_increasing


def test_calibrators_return_valid_probabilities():
    rng = np.random.default_rng(42)
    n = 2000
    y = rng.binomial(1, 0.2, n)
    prob = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.05, n), 0.001, 0.999)

    calibradores = fit_calibrators(y, prob)

    for metodo, calibrador in calibradores.items():
        calibrada = apply_calibrator(metodo, calibrador, prob)
        assert len(calibrada) == n
        assert np.all((calibrada >= 0) & (calibrada <= 1))

    assert np.allclose(logit(np.array([0.5])), 0.0)

    metricas = pd.DataFrame(
        {
            "base": ["validacao", "validacao", "validacao"],
            "metodo": ["original", "platt", "spline"],
            "logloss": [0.50, 0.49, 0.4905],
        }
    )
    assert choose_calibrator(metricas) == "spline"
