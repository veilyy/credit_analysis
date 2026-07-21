"""
train.py - обучение и сохранение артефактов.
 
Артефакты на выходе (пути из config['paths']):
  models/credit_pipeline.pkl     - обученный пайплайн (препроц + энкодер + модель);
  models/segment_thresholds.json - замороженные границы A/B/C.
"""

from __future__ import annotations
  
import joblib
 
from src.data import PROJECT_ROOT, load_config, load_data, split_data
from src.pipeline import build_pipeline

def train(config: dict) -> None:
    # 1. Данные и воспроизводимый сплит.
    df = load_data(config)
    X_train, _, y_train, _ = split_data(df, config)
    print(f"Train: {X_train.shape[0]}")

    # 2. Сборка и обучение пайплайна.
    pipeline = build_pipeline(config)
    pipeline.fit(X_train, y_train)
    print("Пайплайн обучен.")
 
    # 3. Сохранение артефактов
    model_path = PROJECT_ROOT / config["paths"]["model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
 
    joblib.dump(pipeline, model_path)
 
    print(f"Модель сохранена:  {model_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    config = load_config()
    train(config)