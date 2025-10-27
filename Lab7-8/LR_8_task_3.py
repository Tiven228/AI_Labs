import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve
from sklearn.pipeline import make_pipeline

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# --- Функція побудови кривих навчання ---
def plot_learning_curves(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.plot(train_sizes, train_mean, 'o-', label='Тренувальні дані')
    plt.plot(train_sizes, test_mean, 'o-', label='Перевірочні дані')
    plt.title(title)
    plt.xlabel("Кількість тренувальних прикладів")
    plt.ylabel("Оцінка точності (R²)")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Лінійна модель ---
plot_learning_curves(LinearRegression(), X, y, "Криві навчання — лінійна регресія")

# --- Поліноміальна модель (10-й ступінь) ---
poly_model = make_pipeline(PolynomialFeatures(degree=10), LinearRegression())
plot_learning_curves(poly_model, X, y, "Криві навчання — поліноміальна регресія (10 ступінь)")