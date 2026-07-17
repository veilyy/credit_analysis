"""
prediction_logger.py - лог предиктов.

кадждый вызов пишется строкой в jsonl: входные данные + вероятность + 
сегммент + время и верси модели

"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.data import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

def log_predictions(inputs: dict, prob: float, segment: str, model_version: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "prob_default": prob,
        "segment": segment,
        **inputs,  # входные фичи заявки
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")