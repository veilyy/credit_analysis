from fastapi.testclient import TestClient
 
from src.api import app

client = TestClient(app)


VALID_APPLICATION = {
    "loan_amount": 15000,
    "loan_term_days": 30,
    "age": 35,
    "income": 50000,
    "employment_type": "Постоянная",
    "credit_history_bad": 0,
    "prev_loans": 2,
    "region": "Москва",
    "education": "Высшее",
}


def test_health_returns_ok():

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    
    response = client.post("/predict", json= VALID_APPLICATION)

    assert response.status_code == 200

    body = response.json()

    assert 0.0 <= body["prob_default"] <= 1.0 
    assert body["segment"] in {"A", "B", "C"}
    assert body["decision"] in {"approve", "review", "reject"}

    if body["segment"] == "A":
        assert body["decision"] == "approve"
    elif body["segment"] == "B":
        assert body["decision"] == "review"
    else:
        assert body["decision"] == "reject"


