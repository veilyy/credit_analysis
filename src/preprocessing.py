"""
preprocessing.py  препроцессинг как sklearn-трансформер.
 
CreditPreprocessor заполняет пропуски и создаёт инженерные фичи.
Все статистики (медианы) считаются на fit (train) и применяются на transform .
Так нет утечки: параметры заполнения выучены только на train.
"""
from __future__ import annotations
 
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
 
# Категориальные колонки, где пропуск -> отдельная категория "Не указано".
_CATEGORICAL_FILL = ["education", "region", "employment_type"]
 
 
class CreditPreprocessor(BaseEstimator, TransformerMixin):
   
    def __init__(self):
        # Параметры, выучиваемые на fit (train):
        self.age_medians_ = None          # медиана age по employment_type
        self.age_median_global_ = None    # запасная медиана age (для новых категорий)
        self.income_median_ = None        # медиана income
 
    def fit(self, X, y=None):
        X = X.copy()
 
        # employment_type участвует в групповой медиане age -> заполняем ДО groupby,
        emp = X["employment_type"].fillna("Не указано")
 
        self.age_medians_ = (
            X.assign(employment_type=emp).groupby("employment_type")["age"].median()
        )
        self.age_median_global_ = X["age"].median()
        self.income_median_ = X["income"].median()
 
        return self
 
    def transform(self, X):
        X = X.copy()
 
        # 1. Категориальные пропуски -> "Не указано"
        for col in _CATEGORICAL_FILL:
            X[col] = X[col].fillna("Не указано")
 
        # 2. prev_loans, credit_history_bad -> 0
        X["prev_loans"] = X["prev_loans"].fillna(0)
        X["credit_history_bad"] = X["credit_history_bad"].fillna(0)
 
        # 3. age -> медиана по employment_type,
        #    остаток (новые категории) -> глобальная медиана.
        age_filled = X["age"].copy()
        mask = age_filled.isna()
        age_filled[mask] = X.loc[mask, "employment_type"].map(self.age_medians_)
        X["age"] = age_filled.fillna(self.age_median_global_)
 
        # 4. income -> медиана с train.
        X["income"] = X["income"].fillna(self.income_median_)
 
        # --- Новые фичи ---
 
        # 5a. daily_payment - дневная платёжная нагрузка: сумма займа / срок.
        X["daily_payment"] = X["loan_amount"] / X["loan_term_days"].replace(0, np.nan)
        X["daily_payment"] = X["daily_payment"].fillna(0)
 
        # 5b. debt_to_income - доля дохода, которую составляет займ.
        X["debt_to_income"] = X["loan_amount"] / X["income"].replace(0, np.nan)
        X["debt_to_income"] = X["debt_to_income"].fillna(0)
 
        # 5c. payment_to_income - дневной платёж относительно дневного дохода (income/30).
        daily_income = (X["income"] / 30).replace(0, np.nan)
        X["payment_to_income"] = (X["daily_payment"] / daily_income).fillna(0)
 
        # Биннинг
        X["age_group"] = pd.cut(
            X["age"],
            bins=[17, 25, 35, 45, 55, 100],
            labels=["18-24", "25-34", "35-44", "45-54", "55+"],
        )
        X["prev_loans_groups"] = pd.cut(
            X["prev_loans"],
            bins=[-1, 0, 2, 5, float("inf")],
            labels=["0 (новые)", "1-2", "3-5", "6+"],
        )
 
        return X