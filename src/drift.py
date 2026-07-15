"""
drift.py - метрики дрейфа данных: PSI и KS.
 
Пороги PSI:
  < 0.10 - сдвига нет;
  0.10 .. 0.25 - умеренный сдвиг, стоит присмотреться;
  >= 0.25 - значимый сдвиг, модель под угрозой.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_EPS = 1e-6 #малое значение для защиты от log(0)



def psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI для числовых """
    reference, current = reference.dropna(), current.dropna()
    

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    pass


