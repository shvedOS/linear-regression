import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = {
    "area": [30, 35, 40, 45, 50, 55, 60, 65, 70, 75,
             80, 85, 90, 95, 100],
    "price": [3100, 3500, 3900, 4500, 4900, 5400, 6100, 6400, 7100, 7400,
              8100, 8500, 9100, 9600, 10100]
}

df = pd.DataFrame(data)


# print(df.describe())

# df.plot(x="area", y="price")

# plt.show()

def predict(x, w, b):
    return w * x + b


def mse(y_true, y_pred):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)

    return np.mean((y_t - y_p) ** 2)


def gradients(x, y, w, b):
    predictions = predict(x, w, b)
    errors = y - predictions
    dw = -2 * np.mean(x * errors)
    db = -2 * np.mean(errors)

    return dw, db


def train(x, y, w, b, learning_rate, iterations):
    mse_history = []

    for i in range(iterations):
        dw, db = gradients(x, y, w, b)

        w -= learning_rate * dw
        b -= learning_rate * db

        predictions = predict(x, w, b)
        loss = mse(y, predictions)

        mse_history.append((i, w, b, loss))

    return w, b, mse_history


w = 0
b = 0
iterations = 100
learning_rate = 0.0001

# Добавим scaler для сравнения
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df[["area"]]).flatten()

print("\nДо scaling:")
print(df["area"].values)

print("\nПосле scaling:")
print(X_scaled)

# Проверим без scaling
w_original, b_original, history_original = train(df["area"], df["price"], w, b, learning_rate, iterations)

print("\nБез scaling:")
print(f"w = {w_original}")
print(f"b = {b_original}")
print(f"MSE = {history_original[-1][3]}")

# Проверим со scaling
w_scaled, b_scaled, history_scaled = train(X_scaled, df["price"], w, b, 0.1, iterations)

print("\nСо scaling:")
print(f"w = {w_scaled}")
print(f"b = {b_scaled}")
print(f"MSE = {history_scaled[-1][3]}")


iterations_original = [row[0] for row in history_original]
losses_original = [row[3] for row in history_original]

iterations_scaled = [row[0] for row in history_scaled]
losses_scaled = [row[3] for row in history_scaled]


predictions = predict(df["area"], w_original, b_original)

plt.plot(
    iterations_original,
    losses_original,
    label="Без scaling"
)

plt.plot(
    iterations_scaled,
    losses_scaled,
    label="Со scaling"
)
plt.xlabel("Итерация")
plt.ylabel("MSE")
plt.title("Изменение MSE во время обучения")
plt.legend()
plt.grid()
plt.show()

plt.scatter(df["area"], df["price"], label="Данные", color="orange")
plt.plot(df["area"], predictions, label="Линейная регрессия")

plt.xlabel("Площадь")
plt.ylabel("Цена")
plt.title("Линейная регрессия")
plt.legend()
plt.grid()
plt.show()

model = LinearRegression()
X = df[["area"]]
y = df["price"]

model.fit(X, y)

print(f"\nw (sklearn) = {model.coef_[0]}")
print(f"b (sklearn) = {model.intercept_}")

sklearn_predictions = model.predict(X)

print(f"MSE (sklearn) = {mse(y, sklearn_predictions)}")
