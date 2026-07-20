import numpy as np
import pandas as pd

from src.preprocessing import CreditPreprocessor



def make_df(**overrides) -> pd.DataFrame:

    data = {
        "age": [25, 35, 45, 55, 30, 40],
        "income": [30000, 50000, 70000, 90000, 40000, 60000],
        "employment_type": ["Найм", "ИП", "Найм", "ИП", "Найм", "ИП"],
        "credit_history_bad": [0, 1, 0, 1, 0, 1],
        "prev_loans": [0, 1, 2, 3, 4, 5],
        "loan_amount": [10000] * 6,
        "loan_term_days": [30] * 6,
        "region": ["Москва", "СПб"] * 3,
        "education": ["Высшее", "Среднее"] * 3,
    }
    data.update(overrides)

    return pd.DataFrame(data)

def test_fit_learns_medians():
    prep = CreditPreprocessor()

    assert prep.income_median_ == 55000
    assert prep.age_median_global_ is not None
    assert prep.age_medians_ is not None


def test_engineered_features_created():
    result = CreditPreprocessor().fit(make_df()).transform(make_df())

    for col in ["daily_payment", "debt_to_income", "payment_to_income",
                "age_group", "prev_loans_groups"]:
        assert col in result.columns


def test_no_missing_after_transform():
    df = make_df()
    df.loc[0, "income"] = np.nan
    df.loc[1, "age"] = np.nan
    df.loc[2, "education"] = None


    result = CreditPreprocessor().fit(make_df()).transform(df)

    for col in ["income", "age", "prev_loans", "credit_history_bad",
                "daily_payment", "debt_to_income", "payment_to_income"]:
        assert result[col].notna().all(), f"Остались пропуски в {col}"


 