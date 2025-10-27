import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Варіант 3
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

plt.scatter(X, y, color='blue', label='Дані')
plt.title("Згенеровані дані (варіант 3)")
plt.show()

# Лінійна регресія
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

plt.scatter(X, y, color='blue')
plt.plot(X, y_pred_lin, color='red', label='Лінійна регресія')
plt.legend()
plt.show()

# Поліноміальна регресія
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)
y_pred_poly = lin_reg2.predict(X_poly)

plt.scatter(X, y, color='blue')
plt.plot(np.sort(X, axis=0), y_pred_poly[np.argsort(X[:, 0])], color='green', label='Поліноміальна регресія')
plt.legend()
plt.title("Поліноміальна регресія (2-й ступінь)")
plt.show()

print("Лінійна модель: y = {:.2f}x + {:.2f}".format(lin_reg.coef_[0][0], lin_reg.intercept_[0]))
print("Поліноміальна модель: y = {:.2f}x² + {:.2f}x + {:.2f}".format(lin_reg2.coef_[0][1], lin_reg2.coef_[0][0], lin_reg2.intercept_[0]))
