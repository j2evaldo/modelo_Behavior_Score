import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1200

    data = {
        "ID": np.arange(1, n + 1),
        "SAFRA": rng.choice([200510, 200601, 200604, 200605, 200606, 200607], n),
        "LIMIT_BAL": rng.integers(5_000, 200_000, n),
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
        "AGE": rng.integers(21, 70, n),
    }

    for col in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]:
        data[col] = rng.integers(-2, 5, n)

    for col in ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]:
        data[col] = rng.integers(-2_000, 80_000, n)

    for col in ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]:
        data[col] = rng.integers(0, 20_000, n)

    media_limite = data["LIMIT_BAL"] / 100_000
    prob = 1 / (1 + np.exp(-(-1 + 2 * (data["PAY_0"] > 0) - media_limite)))
    data["default"] = rng.binomial(1, prob)

    return pd.DataFrame(data)
