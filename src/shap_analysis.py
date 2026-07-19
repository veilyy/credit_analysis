"""
shap_analysis.py SHAP-интерпретация модели.
 
Для линейной модели SHAP считается через LinearExplainer:
вклад признака = коэффициент x (значение - среднее)
 
Строит два графика (глобальная картина) и печатает разбор одного клиента
(локальное объяснение, что нужно для обоснования отказа).
 
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import shap
 
from src.data import PROJECT_ROOT, load_config, load_data, split_data
 
if __name__ == "__main__":
    config = load_config()
    df = load_data(config)
    X_train, X_test, _, _ = split_data(df, config)
    pipeline = joblib.load(PROJECT_ROOT / config["paths"]["model"])

    # Прогоняем данные через все шаги пайплайна кроме модели:
    X_train_enc = pipeline[:-1].transform(X_train)
    X_test_enc = pipeline[:-1].transform(X_test)
 
    # OneHotEncoder может вернуть разреженную матрицу - SHAP ждёт плотную.
    if sp.issparse(X_train_enc):
        X_train_enc = X_train_enc.toarray()
    if sp.issparse(X_test_enc):
        X_test_enc = X_test_enc.toarray()
 
    names = pipeline.named_steps["encode"].get_feature_names_out()
    model = pipeline.named_steps["model"]
 
    # LinearExplainer: точный SHAP для линейной модели.
    explainer = shap.LinearExplainer(model, X_train_enc)
    shap_values = explainer.shap_values(X_test_enc)
 
    out_dir = PROJECT_ROOT / "png"
    out_dir.mkdir(exist_ok=True)
 
    # 1. Beeswarm: сила, направление и значение признака (цвет)
    shap.summary_plot(shap_values, X_test_enc, feature_names=names, show=False)
    plt.title("SHAP: влияние признаков на риск дефолта")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close()
 
    # 2. Bar: средний модуль вклада (только величина)
    shap.summary_plot(shap_values, X_test_enc, feature_names=names,
                      plot_type="bar", show=False)
    plt.title("SHAP: средний вклад в риск дефолта")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
 
    # 3. Локальное объяснение одного клиента
    i = 0
    base = float(np.ravel(explainer.expected_value)[0])
    total = base + shap_values[i].sum()
 
    print(f"Базовый лог-шанс (средний клиент): {base:+.3f}")
    print(f"Лог-шанс клиента #{i}:             {total:+.3f}")
    print(f"Сдвиг за счёт его признаков:      {total - base:+.3f}\n")
 
    top = sorted(zip(names, shap_values[i]), key=lambda p: abs(p[1]), reverse=True)[:5]
    print(f"Топ-5 факторов решения по клиенту #{i}:")
    for name, val in top:
        direction = "повышает риск" if val > 0 else "снижает риск"
        print(f"  {name:38s} {val:+.3f}  ({direction})")
 
    print(f"\nГрафики сохранены в png/")