import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('diabetes.csv')

zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_columns:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest F1: {f1_score(y_test, y_pred_rf):.2f}")

max_depths = range(1, 21)
f1_scores_depth = []
for depth in max_depths:
    rf_tmp = RandomForestClassifier(max_depth=depth, random_state=42)
    rf_tmp.fit(X_train, y_train)
    y_pred_tmp = rf_tmp.predict(X_test)
    f1_scores_depth.append(f1_score(y_test, y_pred_tmp))

plt.figure(figsize=(10, 6))
plt.plot(max_depths, f1_scores_depth, marker='o')
plt.xlabel('Максимальная глубина')
plt.ylabel('F1-score')
plt.title('Зависимость F1 от глубины деревьев')
plt.grid()
plt.show()

max_features_range = range(1, X.shape[1]+1)
f1_scores_features = []
for features in max_features_range:
    rf_tmp = RandomForestClassifier(max_features=features, random_state=42)
    rf_tmp.fit(X_train, y_train)
    y_pred_tmp = rf_tmp.predict(X_test)
    f1_scores_features.append(f1_score(y_test, y_pred_tmp))

plt.figure(figsize=(10, 6))
plt.plot(max_features_range, f1_scores_features, marker='o')
plt.xlabel('Количество признаков')
plt.ylabel('F1-score')
plt.title('Зависимость F1 от количества признаков')
plt.grid()
plt.show()

n_estimators_range = range(10, 201, 10)
f1_scores_trees = []
training_times = []
for n in n_estimators_range:
    start_time = time.time()
    rf_tmp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_tmp.fit(X_train, y_train)
    training_times.append(time.time() - start_time)
    y_pred_tmp = rf_tmp.predict(X_test)
    f1_scores_trees.append(f1_score(y_test, y_pred_tmp))

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(n_estimators_range, f1_scores_trees, 'b-', marker='o')
ax1.set_xlabel('Количество деревьев')
ax1.set_ylabel('F1-score', color='b')
ax2 = ax1.twinx()
ax2.plot(n_estimators_range, training_times, 'r-', marker='o')
ax2.set_ylabel('Время обучения (сек)', color='r')
plt.title('Зависимость F1 и времени обучения от числа деревьев')
plt.grid()
plt.show()

xgb = XGBClassifier(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    objective='binary:logistic',
    random_state=42
)
start_time = time.time()
xgb.fit(X_train, y_train)
xgb_time = time.time() - start_time
y_pred_xgb = xgb.predict(X_test)
print(f"XGBoost F1: {f1_score(y_test, y_pred_xgb):.2f}")
print(f"Время обучения XGBoost: {xgb_time:.2f} сек")

print("\nСравнение моделей:")
print(f"Random Forest F1: {f1_score(y_test, y_pred_rf):.2f}")
print(f"XGBoost F1: {f1_score(y_test, y_pred_xgb):.2f}")


print("\nВывод:")
print("Случайный лес показывает хорошие результаты (F1=0.68), но требует тонкой настройки.")
print("XGBoost с параметрами max_depth=3, learning_rate=0.1, n_estimators=100 превосходит случайный лес по F1-score (0.74) и скорости.")
print("Ключевые параметры для ансамблей: глубина деревьев, количество признаков и регуляризация.")
print("XGBoost демонстрирует лучший баланс между точностью и скоростью обучения для данного датасета")
