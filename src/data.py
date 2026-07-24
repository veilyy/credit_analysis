"""
Отвечает за три вещи:
  1. load_config  - прочитать config.yaml
  2. load_data    - загрузить сырой CSV
  3. split_data   - воспроизводимый train/test/val split
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(config: dict) -> pd.DataFrame:
    csv_path = PROJECT_ROOT / config["data"]["raw_path"]
    return pd.read_csv(csv_path)


def split_data(df: pd.DataFrame, config: dict):
    
    target = config["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    stratify = y if config["split"].get("stratify", True) else None
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=config["split"]["train_size"],
        stratify=stratify,
        random_state=config["seed"],
    )
    stratify_temp = y_temp if config["split"].get("stratify", True) else None

    X_val, X_test, y_val, y_test = train_test_split(

        X_temp,
        y_temp,
        test_size=0.50,
        random_state=config["seed"],
        stratify=stratify_temp
    )

    return X_train, y_train, X_test, y_test, X_val, y_val 


if __name__ == "__main__":
    cfg = load_config()
    df = load_data(cfg)
    X_train, y_train, X_test, y_test, X_val, y_val = split_data(df, cfg)

    print(f"Данные:        {df.shape[0]} строк, {df.shape[1]} колонок")
    print(f"Train / Test / Val:  {X_train.shape[0]} / {X_test.shape[0]} / {X_val.shape[0]}")
    print(
        f"Доля дефолтов на train: {y_train.mean():.3f} | "
        f"test: {y_test.mean():.3f} | "
        f"val: {y_val.mean():.3f}"
    )
