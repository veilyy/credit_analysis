"""
segmentation.py - сегментация портфеля A/B/C по вероятности дефолта.
 
  A — низкий риск   (score < low_thr)
  B — средний риск  (low_thr <= score < high_thr)
  C — высокий риск  (score >= high_thr)
 
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_thresholds(train_scores: np.ndarray, config:dict) -> dict:
    """
    Считает границы A/B/C по перцентилям train-скоров.
    Возвращает dict
    Квантили берутся из config['segmentation'].
    """
    seg = config["segmentation"]
    low = float(np.quantile(train_scores, seg["low_quantile"]))
    high = float(np.quantile(train_scores, seg["high_quantile"]))

    return {
        "low_thr": low,
        "high_thr": high,
        "low_quantile": seg["low_quantile"],
        "high_quantile": seg["high_quantile"],
    }

def assign_segment(score: float, thresholds: dict) -> str:
    """Относит одну вероятность к сегменту A/B/C по порогам."""
    if score < thresholds["low_thr"]:
        return "A"
    if score < thresholds["high_thr"]:
        return "B"
    return "C"


def assign_segments(scores: np.ndarray, thresholds: dict) -> np.ndarray:
    """Векторная версия assign_segment"""
    low, high = thresholds["low_thr"], thresholds["high_thr"]
    # np.select быстрее списочного цикла и читается как таблица условий.
    conditions = [scores < low, scores < high]
    choices = ["A", "B"]
    return np.select(conditions, choices, default="C")
 
 
def save_thresholds(thresholds: dict, path: str | Path) -> None:
    """Сохраняет пороги в JSON (замораживает артефакт)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
 
 
def load_thresholds(path: str | Path) -> dict:
    """Загружает замороженные пороги (для eval и serving)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 