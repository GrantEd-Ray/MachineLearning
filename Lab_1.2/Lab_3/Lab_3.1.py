import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === 1. Загрузка и первичный анализ Iris ===
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("Названия классов:", iris.target_names)
print("\nПервые строки датасета:")
print(df.head())

# === 2. Визуализация Matplotlib ===
def plot_feature_scatter(x_feature, y_feature, title):
    plt.figure(figsize=(6, 4))
    for target in np.unique(df['target']):
        subset = df[df['target'] == target]
        plt.scatter(subset[x_feature], subset[y_feature], label=iris.target_names[target])
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_feature_scatter('sepal length (cm)', 'sepal width (cm)', 'Sepal: Length vs Width')
plot_feature_scatter('petal length (cm)', 'petal width (cm)', 'Petal: Length vs Width')

# === 3. Seaborn pairplot ===
sns.pairplot(df, hue="target", diag_kind="kde", palette="husl")
plt.suptitle("Seaborn Pairplot Iris", y=1.02)
plt.show()

# === 4. Подготовка двух поднаборов ===
# Поднабор 1: Setosa vs Versicolor
df_sv = df[df['target'].isin([0, 1])]
X_sv = df_sv.drop(columns='target').values
y_sv = df_sv['target'].values

# Поднабор 2: Versicolor vs Virginica
df_vv = df[df['target'].isin([1, 2])]
X_vv = df_vv.drop(columns='target').values
y_vv = df_vv['target'].values

# === 5. Обучение и оценка модели на каждом поднаборе ===
def train_and_evaluate(X, y, title):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{title}")
    print(f"Точность модели: {accuracy:.2f}")
    return clf

clf_sv = train_and_evaluate(X_sv, y_sv, "Setosa vs Versicolor")
clf_vv = train_and_evaluate(X_vv, y_vv, "Versicolor vs Virginica")

# === 6. Генерация собственного набора данных ===
X_gen, y_gen = make_classification(
    n_samples=1000, n_features=2, n_redundant=0,
    n_informative=2, random_state=1, n_clusters_per_class=1
)

# Визуализация
plt.figure(figsize=(6, 4))
plt.scatter(X_gen[:, 0], X_gen[:, 1], c=y_gen, cmap='bwr', edgecolor='k', s=30)
plt.title("Сгенерированный датасет (make_classification)")
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.grid(True)
plt.show()

# === 7. Классификация сгенерированного датасета ===
train_and_evaluate(X_gen, y_gen, "Классификация на make_classification")
