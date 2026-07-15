"""
pipeline.py - сборка обучаемого пайплайна
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import CreditPreprocessor


def build_encoder(config: dict) -> ColumnTransformer:
    
    cat_cols= config["features"]["categorical"]
    num_cols= config["features"]["numeric"]

    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )

def build_model(config: dict) -> LogisticRegression:
    m = config["model"]
    return LogisticRegression(
        C=m["C"],
        penalty=m["penalty"],
        solver=m["solver"],
        class_weight=m["class_weight"],
        max_iter=m["max_iter"],
        random_state=config["seed"],
    )

def build_pipeline(config: dict) -> Pipeline:
    return Pipeline(
        [
            ("preproc", CreditPreprocessor()),
            ("encode", build_encoder(config)),
            ("model", build_model(config)),
        ]
    )