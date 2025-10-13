import numpy as np
from sklearn import preprocessing

#Бінаризація
input_data = np.array([[ 1.3, -3.9,  6.5],
                       [-4.9, -2.2,  1.3],
                       [ 2.2,  6.5, -6.1],
                       [-5.4, -1.4,  2.2]])
data_binarized = preprocessing.Binarizer(threshold=1.1).transform(input_data)
print("\n Binarized data:\n", data_binarized)

#Виключення середнього
print("\nBEFORE: ")
print("Mean: =", input_data.mean(axis=0))
print("Std deviation =", input_data.std(axis=0))

data_scaled = preprocessing.scale(input_data)
print("\nAFTER: ")
print("Mean: =", data_scaled.mean(axis=0))
print("Std deviation =", data_scaled.std(axis=0))

#Масштабування
data_scaled_minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled_minmax = data_scaled_minmax.fit_transform(input_data)
print("\n Min-Max scaled data:\n", data_scaled_minmax)

#Нормалізація
data_normalized_l1 = preprocessing.normalize(input_data, norm='l1')
data_normalized_l2 = preprocessing.normalize(input_data, norm='l2')
print("\nl1 Normalized data:\n", data_normalized_l1)
print("\nl2 Normalized data:\n", data_normalized_l2)
