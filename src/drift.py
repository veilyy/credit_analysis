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
  if len(edges) < 2:
      return 0.0
    
  edges[0], edges[-1] = -np.inf, np.inf

  ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
  cur_pct = np.histogram(current, bins=edges)[0] / len(current)

  ref_pct = np.clip(ref_pct, _EPS, None)
  cur_pct = np.clip(cur_pct, _EPS, None)

  return(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))



    
def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
  """ расчет PSI для категориальных фич """
  cats = sorted(set(reference.dropna()) | set(current.dropna()))

  ref_pct = np.clip([(reference == c).mean() for c in cats], _EPS, None)
  cur_pct = np.clip([(current == c).mean() for c in cats], _EPS, None)

  return(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))



def ks_test(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
  """ KS тест для числовых с p value """
  stat, p_value = ks_2samp(reference.dropna(), current.dropna())
  return float(stat), float(p_value)


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Отчёт по дрейфу: строка на фичу (feature, type, psi, verdict, ks_stat, p_value, drifted).
 
    Списки фич и порог берутся из config. Мониторим только СЫРЫЕ фичи
    """
    threshold = config.get("monitoring", {}).get("psi_significant", 0.25)
 
    numeric = [c for c in config["features"]["numeric"] if c in reference.columns]
    categorical = [c for c in config["features"]["categorical"] if c in reference.columns]
 
    rows = []
    for col in numeric:
        psi = psi_numeric(reference[col], current[col])
        ks_stat, p_value = ks_test(reference[col], current[col])
        rows.append({
            "feature": col, "type": "numeric", "psi": round(psi, 4),
            "ks_stat": round(ks_stat, 4), "p_value": round(p_value, 4),
            "drifted": psi >= threshold,
        })
    for col in categorical:
        psi = psi_categorical(reference[col], current[col])
        rows.append({
            "feature": col, "type": "categorical", "psi": round(psi, 4),
            "ks_stat": np.nan, "p_value": np.nan,
            "drifted": psi >= threshold,
        })
 
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)