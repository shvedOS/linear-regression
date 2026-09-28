import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

w = 100
b = 0
learning_rate = 0.0001


w, b, history = train(df["area"], df["price"], w, b, learning_rate, 100)

predictions = predict(df["area"], w, b)

iterations = [row[0] for row in history]
losses = [row[3] for row in history]


plt.plot(iterations, losses)
plt.xlabel("Итерация")
plt.ylabel("MSE")
plt.title("Изменение MSE во время обучения")
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