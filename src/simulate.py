"""
simulate.py  симуляция дрейфа данных (в учебных рамках).
"""
from __future__ import annotations
 
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score
 
from src.data import PROJECT_ROOT, load_config, load_data, split_data
from src.drift import drift_report
 
 
def shift_feature(df: pd.DataFrame, feature: str, factor: float) -> pd.DataFrame:
    """Возвращает копию df с умноженной на factor фичей."""
    shifted = df.copy()
    shifted[feature] = shifted[feature] * factor
    return shifted
 
 
if __name__ == "__main__":
    config = load_config()
    df = load_data(config)
    X_train, _, X_test, y_test, _, _ = split_data(df, config)
 
    pipeline = joblib.load(PROJECT_ROOT / config["paths"]["model"])
 
    # Демо: сдвигаем income на +40% (например, приходит более обеспеченная аудитория).
    feature, factor = "income", 1.4
    X_shifted = shift_feature(X_test, feature, factor)
 
    # Метрика ДО и ПОСЛЕ сдвига.
    pr_before = average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    pr_after = average_precision_score(y_test, pipeline.predict_proba(X_shifted)[:, 1])
 
    print(f"Сдвиг: {feature} x{factor}\n")
    print("Дрейф (train vs сдвинутый test):")
    print(drift_report(X_train, X_shifted, config).to_string(index=False))
    print()
    print(f"PR-AUC до сдвига:    {pr_before:.4f}")
    print(f"PR-AUC после сдвига: {pr_after:.4f}")
    print(f"Падение:             {pr_before - pr_after:.4f}")