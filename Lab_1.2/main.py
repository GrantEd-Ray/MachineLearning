import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print("Статистика признаков:")
print(df.describe())

feature = 's3'
X_single = df[[feature]].values
y = df['target'].values

reg = LinearRegression()
reg.fit(X_single, y)
y_pred_sklearn = reg.predict(X_single)

print("\n[Sklearn] Коэффициенты:")
print(f"Коэффициент наклона (slope): {reg.coef_[0]:.4f}")
print(f"Свободный член (intercept): {reg.intercept_:.4f}")

def least_squares(x, y):
    x = x.flatten()
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sum((x - mean_x) ** 2)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept

slope_custom, intercept_custom = least_squares(X_single, y)
y_pred_custom = slope_custom * X_single.flatten() + intercept_custom

print("\n[Custom] Коэффициенты:")
print(f"Коэффициент наклона (slope): {slope_custom:.4f}")
print(f"Свободный член (intercept): {intercept_custom:.4f}")

plt.figure(figsize=(10, 5))
plt.scatter(X_single, y, color='blue', label='Данные')
plt.plot(X_single, y_pred_sklearn, color='red', label='Sklearn Linear Regression')
plt.plot(X_single, y_pred_custom, color='green', linestyle='dashed', label='Custom Linear Regression')
plt.xlabel(feature)
plt.ylabel('Target')
plt.title(f'Линейная регрессия по признаку "{feature}"')
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
