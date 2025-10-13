# === ІМПОРТ НЕОБХІДНИХ БІБЛІОТЕК ===
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score,
    precision_score, f1_score, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt


# === ЗАВАНТАЖЕННЯ ВХІДНИХ ДАНИХ ===
df = pd.read_csv('data_metrics.csv')
print(df.head())


# === СТВОРЕННЯ ДОДАТКОВИХ СТОВПЦІВ (ПЕРЕДБАЧЕНІ МІТКИ) ===
thresh = 0.5
df['predicted_RF'] = (df.model_RF >= 0.5).astype('int')
df['predicted_LR'] = (df.model_LR >= 0.5).astype('int')
print(df.head())


# === МАТРИЦЯ ПОМИЛОК (CONFUSION MATRIX) ===
print('Confusion matrix RF:\n', confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print('Confusion matrix LR:\n', confusion_matrix(df.actual_label.values, df.predicted_LR.values))


# === ВЛАСНІ ФУНКЦІЇ ДЛЯ ОБЧИСЛЕННЯ TP, FN, FP, TN ===
def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))

def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))

def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))

def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))


# === ФУНКЦІЇ ДЛЯ ПОВНОЇ МАТРИЦІ ПОМИЛОК ===
def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN

def Smilka_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])

#print(Smilka_confusion_matrix(df.actual_label.values, df.predicted_RF.values))


# === ВЛАСНА ФУНКЦІЯ ДЛЯ ТОЧНОСТІ (ACCURACY) ===
def Smilka_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

print("Accuracy RF:", Smilka_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print("Accuracy LR:", Smilka_accuracy_score(df.actual_label.values, df.predicted_LR.values))


# === ВЛАСНА ФУНКЦІЯ ДЛЯ ПОВНОТИ (RECALL) ===
def Smilka_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)

print("Recall RF:", Smilka_recall_score(df.actual_label.values, df.predicted_RF.values))
print("Recall LR:", Smilka_recall_score(df.actual_label.values, df.predicted_LR.values))


# === ВЛАСНА ФУНКЦІЯ ДЛЯ ТОЧНОСТІ (PRECISION) ===
def Smilka_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)

print("Precision RF:", Smilka_precision_score(df.actual_label.values, df.predicted_RF.values))
print("Precision LR:", Smilka_precision_score(df.actual_label.values, df.predicted_LR.values))


# === ВЛАСНА ФУНКЦІЯ ДЛЯ F1-МІРИ ===
def Smilka_f1_score(y_true, y_pred):
    recall = Smilka_recall_score(y_true, y_pred)
    precision = Smilka_precision_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)

print("F1 RF:", Smilka_f1_score(df.actual_label.values, df.predicted_RF.values))
print("F1 LR:", Smilka_f1_score(df.actual_label.values, df.predicted_LR.values))


# === АНАЛІЗ ВПЛИВУ ЗМІНИ ПОРОГА ===
print('\nScores with threshold = 0.5')
print("Accuracy RF:", Smilka_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print("Recall RF:", Smilka_recall_score(df.actual_label.values, df.predicted_RF.values))
print("Precision RF:", Smilka_precision_score(df.actual_label.values, df.predicted_RF.values))
print("F1 RF:", Smilka_f1_score(df.actual_label.values, df.predicted_RF.values))

print('\nScores with threshold = 0.25')
print("Accuracy RF:", Smilka_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype(int).values))
print("Recall RF:", Smilka_recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype(int).values))
print("Precision RF:", Smilka_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype(int).values))
print("F1 RF:", Smilka_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype(int).values))


# === ПОБУДОВА ROC-КРИВОЇ ТА ОБЧИСЛЕННЯ AUC ===
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k--', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g--', label='perfect')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
