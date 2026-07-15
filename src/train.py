"""
train.py — обучение и сохранение артефактов.
 
Артефакты на выходе (пути из config['paths']):
  models/credit_pipeline.pkl     - обученный пайплайн (препроц + энкодер + модель);
  models/segment_thresholds.json - замороженные границы A/B/C.
"""

from __future__ import annotations
 
from pathlib import Path
 
import joblib
 
from src.data import PROJECT_ROOT, load_config, load_data, split_data
from src.pipeline import build_pipeline
from src.segmentation import fit_thresholds, save_thresholds

def train(config: dict) -> None:
    # 1. Данные и воспроизводимый сплит.
    df = load_data(config)
    X_train, X_test, y_train, y_test = split_data(df, config)
    print(f"Train / Test: {X_train.shape[0]} / {X_test.shape[0]}")

    # 2. Сборка и обучение пайплайна.
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)
    print("Пайплайн обучен.")

    # 3. Пороги сегментов 
    train_scores = pipeline.predict_proba(X_train)[:, 1]
    thresholds = fit_thresholds(train_scores, config)
    print(
        f"Пороги сегментов: A/B = {thresholds['low_thr']:.3f} | "
        f"B/C = {thresholds['high_thr']:.3f}"
    )
 
    # 4. Сохранение артефактов
    model_path = PROJECT_ROOT / config["paths"]["model"]
    thr_path = PROJECT_ROOT / config["paths"]["segment_thresholds"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
 
    joblib.dump(pipeline, model_path)
    save_thresholds(thresholds, thr_path)
 
    print(f"Модель сохранена:  {model_path.relative_to(PROJECT_ROOT)}")
    print(f"Пороги сохранены:  {thr_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    config = load_config()
    train(config)