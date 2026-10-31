import numpy as np
import pandas as pd

from .config import BILL_COLUMNS, PAY_AMT_COLUMNS, PAY_COLUMNS


def create_features(
    base: pd.DataFrame,
    pay_columns: list[str] | None = None,
    bill_columns: list[str] | None = None,
    pay_amt_columns: list[str] | None = None,
) -> pd.DataFrame:
    pay_columns = pay_columns or PAY_COLUMNS
    bill_columns = bill_columns or BILL_COLUMNS
    pay_amt_columns = pay_amt_columns or PAY_AMT_COLUMNS

    return base.assign(
        SUM_BILL=lambda x: x[bill_columns].sum(axis=1),
        AVG_BILL=lambda x: x[bill_columns].mean(axis=1),
        MAX_BILL=lambda x: x[bill_columns].max(axis=1),
        SUM_PAY=lambda x: x[pay_amt_columns].sum(axis=1),
        AVG_PAY=lambda x: x[pay_amt_columns].mean(axis=1),
        MAX_PAY=lambda x: x[pay_amt_columns].max(axis=1),
        UTILIZACAO=lambda x: (x["BILL_AMT1"] / x["LIMIT_BAL"].replace(0, np.nan)).clip(0, 2),
        UTIL_MEDIA=lambda x: (
            x[bill_columns].mean(axis=1) / x["LIMIT_BAL"].replace(0, np.nan)
        ).clip(0, 2),
        PCT_PAGAMENTO=lambda x: (
            x["PAY_AMT1"] / x["BILL_AMT1"].replace(0, np.nan)
        ).clip(0, 2),
        FREQ_ATRASO=lambda x: (x[pay_columns] > 0).sum(axis=1),
        MAX_ATRASO=lambda x: x[pay_columns].max(axis=1),
        LIMIT_PER_CAPITA=lambda x: x["LIMIT_BAL"] / x["AGE"].replace(0, np.nan),
    )
