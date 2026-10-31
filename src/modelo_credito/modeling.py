import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def build_formula(target: str, features: list[str]) -> str:
    return f"{target} ~ " + " + ".join(features)


def fit_logit_formula(data: pd.DataFrame, formula: str, disp: int = 0):
    return smf.logit(formula, data=data).fit(disp=disp)


def stepwise_logit(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    threshold: float = 0.05,
) -> tuple[list[str], object, pd.DataFrame]:
    selecionadas = features.copy()
    historico = []

    while True:
        formula = build_formula(target, selecionadas)
        modelo = fit_logit_formula(data, formula)
        pvalues = modelo.pvalues.drop("Intercept")
        maior_p = pvalues.max()

        if maior_p <= threshold:
            break

        removida = pvalues.idxmax()
        historico.append(
            {
                "variavel_removida": removida,
                "p_value": maior_p,
                "aic": modelo.aic,
                "bic": modelo.bic,
            }
        )
        selecionadas.remove(removida)

    return selecionadas, modelo, pd.DataFrame(historico)


def coefficient_table(model) -> pd.DataFrame:
    tabela = pd.DataFrame(
        {
            "variavel": model.params.index,
            "coef": model.params.values,
            "p_value": model.pvalues.values,
            "std_err": model.bse.values,
            "z": model.tvalues.values,
        }
    ).sort_values("p_value", ascending=False)
    tabela["significativo_5pct"] = tabela["p_value"] < 0.05
    return tabela.reset_index(drop=True)


def predict_logit(model, data: pd.DataFrame) -> np.ndarray:
    return model.predict(data).values


def model_to_portable(model, features: list[str]) -> dict[str, object]:
    return {
        "formula": model.model.formula,
        "variaveis": features,
        "intercepto": float(model.params["Intercept"]),
        "coeficientes": {var: float(model.params[var]) for var in features},
    }


def save_portable_model(
    model,
    features: list[str],
    path: str | Path,
) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    modelo = model_to_portable(model, features)

    with open(path, "wb") as file:
        pickle.dump(modelo, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(path.with_suffix(".json"), "w") as file:
        json.dump(modelo, file, indent=2)

    return modelo


def load_portable_model(path: str | Path) -> dict[str, object]:
    with open(path, "rb") as file:
        return pickle.load(file)


def predict_portable(data: pd.DataFrame, portable_model: dict[str, object]) -> np.ndarray:
    variaveis = portable_model["variaveis"]
    X = np.column_stack(
        [np.ones(len(data))] + [data[var].values for var in variaveis]
    )
    beta = np.array(
        [portable_model["intercepto"]]
        + [portable_model["coeficientes"][var] for var in variaveis]
    )
    return 1 / (1 + np.exp(-X @ beta))
