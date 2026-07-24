from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import streamlit as st
from sklearn.metrics import average_precision_score

from src.data import PROJECT_ROOT, load_config, load_data, split_data
from src.drift import drift_report
from src.simulate import shift_feature


@st.cache_data
def get_data():
    config = load_config()
    df = load_data(config)
    X_train, _, X_test, y_test, _, _ = split_data(df, config)
    return config, X_train, X_test, y_test

@st.cache_resource
def get_model(model_path):
    return joblib.load(model_path)


config, X_train, X_test, y_test = get_data()
pipeline = get_model(PROJECT_ROOT / config["paths"]["model"])


st.set_page_config(page_title="Мониторинг скоринга", layout="wide")
st.title("Мониторинг кредитной скоринг-модели")


# --- Панель управления: симуляция дрейфа ---
st.sidebar.header("Симуляция дрейфа")
numeric_raw = [c for c in config["features"]["numeric"] if c in X_test.columns]
feature = st.sidebar.selectbox("Фича для сдвига", numeric_raw, index=numeric_raw.index("income"))
factor = st.sidebar.slider("Сила сдвига (x)", 1.0, 2.0, 1.0, 0.05)
st.sidebar.caption("1.0 = данные не тронуты. Двигай вправо - распределение фичи уедет.")


# Текущий батч = test со сдвинутой фичей.
X_current = shift_feature(X_test, feature, factor)
 

# --- Метрики качества до/после ---
pr_before = average_precision_score(y_test, pipeline.predict_proba(X_test)[:, 1])
pr_after = average_precision_score(y_test, pipeline.predict_proba(X_current)[:, 1])
 
c1, c2, c3 = st.columns(3)
c1.metric("PR-AUC (baseline)", f"{pr_before:.4f}")
c2.metric("PR-AUC (текущий батч)", f"{pr_after:.4f}", f"{pr_after - pr_before:+.4f}")
c3.metric("Сдвиг", f"{feature} x{factor}")


# --- Сигнал переобучения ---
report = drift_report(X_train, X_current, config)
drifted = report[report["drifted"]]["feature"].tolist()
 

if drifted:
    st.error(f"Обнаружен значимый дрейф: {', '.join(drifted)}. Рекомендуется переобучение.")
else:
    st.success("Дрейф в пределах нормы. Модель работает на ожидаемых данных.")
 

# --- Таблица дрейфа + график PSI ---
left, right = st.columns([3, 2])
with left:
    st.subheader("Отчёт по дрейфу (PSI / KS)")
    st.dataframe(report, use_container_width=True, hide_index=True)
with right:
    st.subheader("PSI по фичам")
    st.bar_chart(report.set_index("feature")["psi"])
 
st.caption(
    "PSI: <0.10 нет сдвига · 0.10–0.25 умеренный · ≥0.25 значимый (флаг drifted). "
)