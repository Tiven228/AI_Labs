import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Завантаження даних
data = np.loadtxt('data_multivar_regr.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Лінійна регресія
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_pred_linear = linear_regressor.predict(X_test)

print("\nLinear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred_linear), 2))
print("R2 score =", round(sm.r2_score(y_test, y_pred_linear), 2))

# Поліноміальна регресія
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# Перевірка точки
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

print("\nLinear regression prediction:", linear_regressor.predict(datapoint))
print("Polynomial regression prediction:", poly_linear_model.predict(poly_datapoint))
