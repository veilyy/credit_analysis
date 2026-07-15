import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CreditPreprocessor(BaseEstimator, TransformerMixin):
    """
    Препроцессинг МФО
    Все статистики (медианы) считаются на fit (train) и применяются на transform (test).
    """

    def __init__(self):
        # Сюда fit запишет выученные на train параметры
        self.age_medians_ = None      # медиана age по employment_type
        self.age_median_global_ = None
        self.income_median_ = None

    def fit(self, X, y=None):
        X = X.copy()

        # employment_type нужен для групповой медианы age — заполняем его «Не указано»
        # ДО расчёта медиан (иначе NaN-группа сломает groupby)
        emp = X['employment_type'].fillna('Не указано')

        # Медиана возраста по типу занятости
        self.age_medians_ = X.assign(employment_type=emp).groupby('employment_type')['age'].median()
        self.age_median_global_ = X['age'].median()

        # Медиана дохода - выученный параметр
        self.income_median_ = X['income'].median()

        return self

    def transform(self, X):
        X = X.copy()

        # --- 1. Категориальные пропуски → фиксированное правило «Не указано» (утечки нет) ---
        for col in ['education', 'region', 'employment_type']:
            X[col] = X[col].fillna('Не указано')

        # --- 2. prev_loans, credit_history_bad → фиксированное правило: 0 ---
        X['prev_loans'] = X['prev_loans'].fillna(0)
        X['credit_history_bad'] = X['credit_history_bad'].fillna(0)

        # --- 3. age → медиана по employment_type (параметр с train) ---
        # map по группе; где группы не было в train — общий медиан
        age_filled = X['age'].copy()
        mask = age_filled.isna()
        age_filled[mask] = X.loc[mask, 'employment_type'].map(self.age_medians_)
        X['age'] = age_filled.fillna(self.age_median_global_)  # подстраховка

        # --- 4. income → медиана с train ---
        X['income'] = X['income'].fillna(self.income_median_)

        # --- 5. Биннинг → фиксированные границы (утечки нет) ---
        X['age_group'] = pd.cut(
            X['age'],
            bins=[17, 25, 35, 45, 55, 100],
            labels=['18-24', '25-34', '35-44', '45-54', '55+']
        )
        X['prev_loans_groups'] = pd.cut(
            X['prev_loans'],
            bins=[-1, 0, 2, 5, float('inf')],
            labels=['0 (новые)', '1-2', '3-5', '6+']
        )

        return X