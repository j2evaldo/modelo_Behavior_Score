import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.calibration import calibration_curve as sklearn_calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


def logit(probabilities: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(probabilities, eps, 1 - eps)
    return np.log(p / (1 - p))


def expected_calibration_error(
    y: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> float:
    limites = np.unique(np.quantile(probabilities, np.linspace(0, 1, bins + 1)))
    indices = np.digitize(probabilities, limites[1:-1])
    erro = 0.0

    for i in range(len(limites) - 1):
        mask = indices == i
        if mask.sum() == 0:
            continue
        erro += mask.mean() * abs(y[mask].mean() - probabilities[mask].mean())

    return float(erro)


def calibration_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "brier": float(brier_score_loss(y, probabilities)),
        "logloss": float(log_loss(y, probabilities)),
        "auc": float(roc_auc_score(y, probabilities)),
        "ece": expected_calibration_error(y, probabilities),
        "prob_media": float(probabilities.mean()),
        "bad_rate": float(y.mean()),
    }


def oof_predictions(
    data: pd.DataFrame,
    formula: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prob_oof = np.zeros(len(data))

    for idx_treino, idx_validacao in kf.split(data):
        modelo = smf.logit(formula, data=data.iloc[idx_treino]).fit(disp=0)
        prob_oof[idx_validacao] = modelo.predict(data.iloc[idx_validacao]).values

    return prob_oof


def fit_calibrators(
    y: np.ndarray,
    prob_oof: np.ndarray,
    n_knots: int = 5,
    degree: int = 3,
) -> dict[str, object]:
    z_oof = logit(prob_oof).reshape(-1, 1)

    calibrador_platt = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=1000,
    ).fit(z_oof, y)

    calibrador_isotonic = IsotonicRegression(out_of_bounds="clip").fit(prob_oof, y)

    calibrador_spline = make_pipeline(
        SplineTransformer(n_knots=n_knots, degree=degree, include_bias=False),
        LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000),
    ).fit(z_oof, y)

    return {
        "platt": calibrador_platt,
        "isotonic": calibrador_isotonic,
        "spline": calibrador_spline,
    }


def apply_calibrator(
    method: str,
    calibrator,
    probabilities: np.ndarray,
) -> np.ndarray:
    if method == "original":
        return probabilities

    if method == "isotonic":
        return calibrator.predict(probabilities)

    z = logit(probabilities).reshape(-1, 1)
    return calibrator.predict_proba(z)[:, 1]


def choose_calibrator(
    metrics: pd.DataFrame,
    tolerance: float = 0.001,
    prefer: str = "spline",
) -> str:
    validacao = metrics.query("base == 'validacao'")
    validacao = validacao[validacao["metodo"] != "original"]
    melhor_logloss = validacao["logloss"].min()
    candidatos = validacao[validacao["logloss"] <= melhor_logloss + tolerance]

    if prefer in candidatos["metodo"].values:
        return prefer
    return candidatos.sort_values("logloss").iloc[0]["metodo"]


def save_calibrator(
    calibrator,
    method: str,
    cutoff: float,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(
            {
                "metodo": method,
                "calibrador": calibrator,
                "corte_calibrado": cutoff,
            },
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def load_calibrator(path: str | Path) -> dict[str, object]:
    with open(path, "rb") as file:
        return pickle.load(file)


def calibration_curve(
    y: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> tuple[np.ndarray, np.ndarray]:
    return sklearn_calibration_curve(
        y,
        probabilities,
        n_bins=n_bins,
        strategy=strategy,
    )
