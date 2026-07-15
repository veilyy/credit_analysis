"""
eval.py оценка обученной модели на тесте.
 
Загружает артефакты (модель + пороги), считает метрики на том же
воспроизводимом сплите, что и train.py, печатает отчёт и сохраняет metrics.json.
 
Метрики:
  PR-AUC -> мейн (несбалансированные классы, важен класс дефолта);
  ROC-AUC -> вспомогательная;
  Gini -> 2*ROC_AUC - 1, стандарт в кредитном скоринге;
  classification_report -> precision/recall по классам;
  default rate по сегментам A/B/C -> проверка, что сегментация разделяет риск.
 
"""
from __future__ import annotations
 
import json
 
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
 
from src.data import PROJECT_ROOT, load_config, load_data, split_data
from src.segmentation import assign_segments, load_thresholds
 
 
def evaluate(config: dict) -> dict:
    # 1. Тот же сплит
    df = load_data(config)
    _, X_test, _, y_test = split_data(df, config)
 
    # 2. Загрузка артефактов
    model_path = PROJECT_ROOT / config["paths"]["model"]
    thr_path = PROJECT_ROOT / config["paths"]["segment_thresholds"]
    pipeline = joblib.load(model_path)
    thresholds = load_thresholds(thr_path)
 
    # 3. Предсказания
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
 
    # 4. Метрики качества
    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    gini = (2 * roc_auc) - 1
 
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"Gini:     {gini:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["no_default", "default"]))
 
    # 5. Default rate по сегментам - проверка, что A/B/C реально разделяют риск.
    segments = assign_segments(y_proba, thresholds)
    seg_df = pd.DataFrame({"segment": segments, "default": y_test.values})
    seg_summary = (
        seg_df.groupby("segment")["default"]
        .agg(clients="size", default_rate="mean")
        .reindex(["A", "B", "C"])
    )
    print("Default rate по сегментам:")
    print(seg_summary.round(3).to_string())
 
    # 6. Сохранение metrics.json
    metrics = {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "gini": round(float(gini), 4),
        "segments": {
            seg: {
                "clients": int(seg_summary.loc[seg, "clients"]),
                "default_rate": round(float(seg_summary.loc[seg, "default_rate"]), 4),
            }
            for seg in ["A", "B", "C"]
        },
    }
    metrics_path = PROJECT_ROOT / config["paths"]["metrics"]
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nМетрики сохранены в: {metrics_path.relative_to(PROJECT_ROOT)}")
 
    return metrics
 
 
if __name__ == "__main__":
    config = load_config()
    evaluate(config)