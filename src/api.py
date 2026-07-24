"""
api.py - сервис скоринга на FastAPI.
 
Сервисная дверь системы: принимает заявку, возвращает вероятность
дефолта и сегмент A/B/C.
"""

from __future__ import annotations

import json

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.data import PROJECT_ROOT, load_config
from src.segmentation import assign_segment, load_thresholds

# --- Загрузка артефактов при старте ---

config = load_config()
pipeline = joblib.load(PROJECT_ROOT / config["paths"]["model"])
thresholds = load_thresholds(PROJECT_ROOT / config["paths"]["thresholds"])

# пороги максимизирующие прибыль портфеля
with open(PROJECT_ROOT / "artifacts" / "thresholds.json", encoding="utf-8") as f:
    policy = json.load(f)

app = FastAPI()



class LoanApplication(BaseModel):
    """Pydantic валидиация"""

    loan_amount: float | None
    loan_term_days: float | None
    age: float | None = None
    income: float | None = None
    employment_type: str | None = None
    credit_history_bad: float | None = None
    prev_loans: float | None = None
    region: str | None = None
    education: str | None = None
 

class PredictionResponse(BaseModel):
    prob_default: float
    segment: str
    decision: str



def make_decision(prob: float) -> str:
    
    if prob < policy["low_thr"]:
        return "approve"
    if prob < policy["high_thr"]:
        return "review"
    return "reject"


@app.get("/health")
def health():
    
    return {
        "status": "ok",
        "model_version": app.version,
        "policy": {"Порог утверждения": policy["low_thr"], "Порог проверки": policy["high_thr"]},
    }
 

@app.post("/predict", response_model=PredictionResponse)

def predict(application: LoanApplication) -> PredictionResponse:
    
    row = pd.DataFrame([application.model_dump()])

    prob = float(pipeline.predict_proba(row)[:, 1][0])
    segment = assign_segment(prob, thresholds)
    decision = make_decision(prob)


    return PredictionResponse(prob_default=round(prob, 4), segment=segment, decision=decision)