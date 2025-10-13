import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utilities import visualize_classifier  # твоя функція для візуалізації

# Завантаження даних
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Створення і тренування класифікатора
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)

# Прогнозування на тестових даних
y_test_pred = classifier_new.predict(X_test)

# Обчислення точності
accuracy = 100.0 * accuracy_score(y_test, y_test_pred)
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

# Візуалізація класифікації
# Якщо більше ніж 2 ознаки, беремо лише перші 2 для графіку
X_test_vis = X_test[:, :2]
visualize_classifier(classifier_new, X_test_vis, y_test)

# Крос-валідація (3-fold) для оцінки метрик
num_folds = 3
classifier_cv = GaussianNB()  # новий об'єкт для крос-валідації

accuracy_values = cross_val_score(classifier_cv, X, y, scoring='accuracy', cv=num_folds)
print("Cross-validated Accuracy: " + str(round(100 * accuracy_values.mean(), 2)) + "%")

precision_values = cross_val_score(classifier_cv, X, y, scoring='precision_weighted', cv=num_folds)
print("Cross-validated Precision: " + str(round(100 * precision_values.mean(), 2)) + "%")

recall_values = cross_val_score(classifier_cv, X, y, scoring='recall_weighted', cv=num_folds)
print("Cross-validated Recall: " + str(round(100 * recall_values.mean(), 2)) + "%")

f1_values = cross_val_score(classifier_cv, X, y, scoring='f1_weighted', cv=num_folds)
print("Cross-validated F1: " + str(round(100 * f1_values.mean(), 2)) + "%")

data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
# Створення наївного байєсовського класифікатора
classifier = GaussianNB()
# Тренування класифікатора
classifier.fit(X, y)
# Прогнозування значень для тренувальних даних
y_pred = classifier.predict(X)
# Обчислення якості класифікатора
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("\nAccuracy of Naive Bayes classifier =", round(accuracy, 2), "%")
# Візуалізація результатів роботи класифікатора
visualize_classifier(classifier, X, y)