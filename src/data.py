"""
Отвечает за три вещи:
  1. load_config  - прочитать config.yaml
  2. load_data    - загрузить сырой CSV
  3. split_data   - воспроизводимый train/test split
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Читает конфиг в словарь."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    """
    Загружает сырой CSV. Путь берётся из config и резолвится от корня проекта
    """
    csv_path = PROJECT_ROOT / config["data"]["raw_path"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Не найден файл данных: {csv_path}\n"
            f"Проверь config['data']['raw_path'] (сейчас: {config['data']['raw_path']})."
        )
    return pd.read_csv(csv_path)


def split_data(df: pd.DataFrame, config: dict):
    """
    Делит данные на train/test.
    """
    target = config["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if config["split"].get("stratify", True) else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config["split"]["test_size"],
        stratify=stratify,
        random_state=config["seed"],
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    cfg = load_config()
    df = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    print(f"Данные:        {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Train / Test:  {X_train.shape[0]} / {X_test.shape[0]}")
    print(
        f"Доля дефолтов на train: {y_train.mean():.3f} | "
        f"test: {y_test.mean():.3f}"
    )
