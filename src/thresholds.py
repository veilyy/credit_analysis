"""
thresholds.py  подбор двух порогов, максимизирующих прибыль портфеля.

прибыль расчитывается с функции total_profit с модуля profit.py
 
"""

import json
 
import joblib
import numpy as np
 
from src.profit import total_profit
from src.data import PROJECT_ROOT, load_config, load_data, split_data

def find_best_thresholds(y, proba, loan, biz, step = 0.02):
    
    grid = np.arange(0, 1 + step, step)
    max_share= biz.get("max_review_share", 1.0)
    best = {"t_low": 0.0, "t_high": 0.0, "profit": -np.inf}

    for t_low in grid:
        for t_high in grid:
            if t_high < t_low:
                continue
            
            # доля заявок, попавших в серую зону
            review_share = ((proba >= t_low) & (proba < t_high)).mean()
            if review_share > max_share:
                continue


            profit = total_profit(y, proba, loan, t_low, t_high, biz)
            if profit > best["profit"]:
                best = {"t_low": float(t_low), "t_high": float(t_high), "profit": profit}

    return best



if __name__ == "__main__":
    config = load_config()
    biz = config["business"]
 
    # Данные и модель
    df = load_data(config)
    _, X_test, _, y_test = split_data(df, config)
    pipeline = joblib.load(PROJECT_ROOT / config["paths"]["model"])
 
    proba = pipeline.predict_proba(X_test)[:, 1]
    loan = X_test["loan_amount"].values
    y = y_test.values
 
    # Поиск оптимума
    best = find_best_thresholds(y, proba, loan, biz)
 
    # Сравнение с наивными стратегиями
    profit_05 = total_profit(y, proba, loan, 0.5, 0.5, biz)
    profit_all = total_profit(y, proba, loan, 1.0, 1.0, biz)
 
    print(f"Оптимальные пороги: t_low = {best['t_low']:.2f} | t_high = {best['t_high']:.2f}")
    print(f"Прибыль (оптимум):     {best['profit']:>15,.0f} руб")
    print(f"Прибыль (порог 0.5):   {profit_05:>15,.0f} руб")
    print(f"Прибыль (одобрить всех):{profit_all:>14,.0f} руб")
    print(f"Выигрыш против 0.5:    {best['profit'] - profit_05:>15,.0f} руб")
 
    # Сохранение результата
    out_path = PROJECT_ROOT / "models" / "policy_thresholds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
 
    print(f"\nСохранено: {out_path.relative_to(PROJECT_ROOT)}")