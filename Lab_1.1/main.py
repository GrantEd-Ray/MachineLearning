import csv
import matplotlib.pyplot as plt

hours, hours_data = [], []
scores, scores_data = [], []

with open("student_scores.csv", 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        hours.append(row[0])
        scores.append(row[1])

print(hours[0])
for i in range(1, len(hours)):
    hours_data.append(float(hours[i]))
    print(hours[i], end=" ")
print()

print(scores[0])
for i in range(1, len(scores)):
    scores_data.append(int(scores[i]))
    print(scores[i], end=" ")
print()
print()

print("Название:", hours[0])
print("Кол-во:", len(hours_data))
print("Мин:", min(hours_data))
print("Макс:", max(hours_data))
print("Среднее:", sum(hours_data) / len(hours_data))
print()

print("Название:", scores[0])
print("Кол-во:", len(scores_data))
print("Мин:", min(scores_data))
print("Макс:", max(scores_data))
print("Среднее:", sum(scores_data) / len(scores_data))
print()

print(f"Какие данные взять в качестве X?\n1. {hours[0]}\n2. {scores[0]}")
while True:
    choice = input("Ваш выбор: ")
    if choice == "1":
        X = hours_data
        Y = scores_data
        break
    elif choice == "2":
        X = scores_data
        Y = hours_data
        break
    else:
        print("Повторите ещё раз.")

x_avg = sum(X) / len(X)
y_avg = sum(Y) / len(Y)

s_xy = 0
s_x2 = 0

for i in range(len(X)):
    s_xy += (X[i] - x_avg) * (Y[i] - y_avg)
    s_x2 += pow(X[i] - x_avg, 2)

b_1 = s_xy / s_x2
b_0 = y_avg - b_1 * x_avg

f = lambda x: b_0 + b_1 * x

Y_reg = []
for x in X:
    Y_reg.append(f(x))

fig, ax = plt.subplots(1)

for i in range(len(X)):
    rect = plt.Rectangle((X[i], f(X[i])), (-Y[i] + f(X[i])), Y[i] - f(X[i]), linewidth=1, edgecolor='r', hatch='//////', facecolor='none')
    ax.add_patch(rect)

plt.scatter(X, Y)
plt.plot(X, Y_reg, color='red')
plt.axis('equal')
plt.show()
