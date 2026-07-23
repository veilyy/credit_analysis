FROM python:3.13-slim

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/

RUN python -m src.train && python -m src.thresholds

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
