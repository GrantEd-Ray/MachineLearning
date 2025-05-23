import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc
import seaborn as sns

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Titanic.csv")

df_clean = df.dropna()

cols_to_drop = [col for col in df_clean.columns if df_clean[col].dtype == 'object' and col not in ['Sex', 'Embarked']]
df_clean = df_clean.drop(columns=cols_to_drop)

df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
df_clean['Embarked'] = df_clean['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

df_clean = df_clean.drop(columns=[col for col in ['PassengerId'] if col in df_clean.columns])

percent_lost = (1 - len(df_clean) / len(df)) * 100
print(f"Потеряно данных: {percent_lost:.2f}%")

X = df_clean.drop(columns='Survived')
y = df_clean['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

X_no_embarked = X.drop(columns='Embarked')
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_no_embarked, y, test_size=0.3, random_state=42)
clf2 = LogisticRegression(max_iter=1000)
clf2.fit(X_train2, y_train2)
y_pred2 = clf2.predict(X_test2)
accuracy2 = accuracy_score(y_test2, y_pred2)
print(f"Точность без Embarked: {accuracy2:.2f}")

cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Precision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}")

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

y_scores = clf.predict_proba(X_test)[:, 1]
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_scores)
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.plot(recall_curve, precision_curve)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve")
plt.grid(True)
plt.show()

plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_scores_svm = svm_model.predict_proba(X_test)[:, 1]

print("SVM")
print(f"Precision: {precision_score(y_test, y_pred_svm):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_svm):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_svm):.2f}")

knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
y_scores_knn = knn_model.predict_proba(X_test)[:, 1]

print("KNN")
print(f"Precision: {precision_score(y_test, y_pred_knn):.2f}")
print(f"Recall: {recall_score(y_test, y_pred_knn):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_knn):.2f}")
print("\nДля выбора наилучшей модели — логистической регрессии, SVM и KNN — опираемся на следующие метрики классификации:\n1. Точность (Precision) — доля правильно предсказанных положительных классов из всех предсказанных как положительные.\n2. Полнота (Recall) — доля правильно предсказанных положительных классов из всех реальных положительных.\n3. F1-мера — гармоническое среднее между precision и recall, отражает баланс между ними.\nДополнительно можно учитывать ROC AUC (площадь под ROC-кривой) и визуальный анализ кривых.")
