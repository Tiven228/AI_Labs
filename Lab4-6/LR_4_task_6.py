# === ІМПОРТ НЕОБХІДНИХ БІБЛІОТЕК ===
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utilities import visualize_classifier  # з попередньої лабораторної

# === ЗАВАНТАЖЕННЯ ДАНИХ ===
input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# === РОЗБИТТЯ НА ТРЕНУВАЛЬНІ І ТЕСТОВІ ДАНІ ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# === СТВОРЕННЯ ТА НАВЧАННЯ МОДЕЛІ SVM ===
# Використовуємо ядро RBF (радіальне), яке добре працює для нелінійних меж
classifier_svm = svm.SVC(kernel='rbf', gamma='auto')
classifier_svm.fit(X_train, y_train)

# === ПРОГНОЗ НА ТЕСТОВИХ ДАНИХ ===
y_pred = classifier_svm.predict(X_test)

# === РОЗРАХУНОК ПОКАЗНИКІВ ЯКОСТІ ===
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, average='macro')
prec = precision_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("=== Показники якості моделі SVM ===")
print(f"Accuracy: {acc:.3f}")
print(f"Recall: {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1-score: {f1:.3f}")

# === ВІЗУАЛІЗАЦІЯ КЛАСИФІКАЦІЇ ===
visualize_classifier(classifier_svm, X, y)
