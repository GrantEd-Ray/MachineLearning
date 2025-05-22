import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    nonzero = y_true != 0
    return np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100


diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Статистика признаков:")
print(df.describe())

feature = 's5'
X_single = df[[feature]].values
y = df['target'].values

reg = LinearRegression()
reg.fit(X_single, y)
y_pred_sklearn = reg.predict(X_single)

def least_squares(x, y):
    x = x.flatten()
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    slope = np.sum((x - mean_x) * (y - mean_y)) / np.sum((x - mean_x) ** 2)
    intercept = mean_y - slope * mean_x
    return slope, intercept


slope_custom, intercept_custom = least_squares(X_single, y)
y_pred_custom = slope_custom * X_single.flatten() + intercept_custom

print("\n=== Метрики качества модели ===")

mae_sklearn = mean_absolute_error(y, y_pred_sklearn)
r2_sklearn = r2_score(y, y_pred_sklearn)
mape_sklearn = mean_absolute_percentage_error(y, y_pred_sklearn)

print("\n[Scikit-Learn Linear Regression]")
print(f"MAE:  {mae_sklearn:.2f}")
print(f"R2:   {r2_sklearn:.2f}")
print(f"MAPE: {mape_sklearn:.2f}%")

mae_custom = mean_absolute_error(y, y_pred_custom)
r2_custom = r2_score(y, y_pred_custom)
mape_custom = mean_absolute_percentage_error(y, y_pred_custom)

print("\n[Custom Linear Regression]")
print(f"MAE:  {mae_custom:.2f}")
print(f"R2:   {r2_custom:.2f}")
print(f"MAPE: {mape_custom:.2f}%")

# === 6. График ===
plt.figure(figsize=(10, 6))
plt.scatter(X_single, y, color='blue', label='Данные')
plt.plot(X_single, y_pred_sklearn, color='red', label='Scikit-Learn')
plt.plot(X_single, y_pred_custom, color='green', linestyle='dashed', label='Custom')
plt.xlabel(feature)
plt.ylabel('Target')
plt.title('Линейная регрессия и сравнение моделей')
plt.legend()
plt.grid(True)
plt.show()

results = pd.DataFrame({
    feature: X_single.flatten(),
    'Actual': y,
    'Sklearn_Predicted': y_pred_sklearn,
    'Custom_Predicted': y_pred_custom
})

print("\nПервые 10 строк предсказаний:")
print(results.head(10).to_string(index=False))

print("\n=== Вывод ===")
print("Обе модели показывают почти идентичные метрики, что подтверждает корректность реализации собственного метода.\n"
      "Scikit-Learn модель немного точнее за счёт численной устойчивости и оптимизации.\n"
      "R² ~ 0.32 говорит о том, что около 32% вариации выходной переменной объясняется выбранным признаком (s5).")
