"""
profit.py - анализ профита с учетом щатрат на review и разного рода ошибок.
 
Политика трёх действий по двум порогам вероятности дефолта p:
    p < t_low            -> APPROVE 
    t_low <= p < t_high  -> REVIEW 
    p >= t_high          -> REJECT 

Экономические параметры берутся из конфига.
 
"""
from __future__ import annotations

import numpy as np
import pandas as pd

def assign_actions(proba: np.ndarray, t_low: float, t_hig: float) -> np.ndarray:     
    """Функция относит вероятности к категориям approve / review / reject."""

    return np.where(proba < t_low, "approve",
                np.where(proba < t_high, "review", "reject"))


def payoff_per_client(
    y_true: np.ndarray,
    proba: np.ndarray,
    loan_amount: np.ndarray,
    t_low: float,
    t_high: float,
    biz: dict,
) -> np.ndarray:
    """
    Денежный итог по каждому клиенту.
    Возвращает эррей выплат.

    """

    y = np.asarray(y_true)
    L = np.asarray(loan_amount)
    actions = assign_actions(np.asarray(proba), t_low, t_high)

    profit = biz["profit_rate"]
    loss = biz["loss_rate"]
    cost = biz["manual_review_cost"]
    catch = biz["review_catch_rate"]

    pay = np.zeros(len(y), dtype=float)

    good = y == 0
    bad = y == 1

    # ОДОБРЕНИЕ: хороший -> +profit*L ; дефолт -> -loss*L
    pay[(action == "approve") & good] = profit * L[(action == "approve") & good]
    pay[(action == "approve") & bad] = -loss * L[(action == "approve") & bad]

    # РУЧНАЯ ПРОВЕРКА: хороший -> +profit*L ; дефолт -> -(1-catch)*loss*L
    pay[(action == "review") & good] = profit * L[(action == "review") & good] - cost
    pay[(action == "review") & bad] = -(1 - catch) * loss * L[(action == "review") & bad] - cost

    return pay



def total_profit(
    y_true, proba, loan_amount, t_low: float, t_high: float, biz: dict
) -> float:
    """Суммарная ожидаемая прибыль портфеля."""
    return float(payoff_per_client(y_true, proba, loan_amount, t_low, t_high, biz).sum())