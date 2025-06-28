import os

import cv2

os.environ["KERAS_BACKEND"] = "torch"
from keras.datasets.cifar10 import load_data as cifar
import keras.layers
import numpy as np


# [0, 1, 4, 6, 7, 8, 9]
def loadCifar(resize=0):
    # [0, 1, 4, 6, 7, 8, 9]
    (X_train, y_train), (X_test, y_test) = cifar()

    class_0_index = np.where(y_train.reshape(-1) == 2)
    class_1_index = np.where(y_train.reshape(-1) == 3)
    class_2_index = np.where(y_train.reshape(-1) == 5)
    class_0_index_test = np.where(y_test.reshape(-1) == 2)
    class_1_index_test = np.where(y_test.reshape(-1) == 3)
    class_2_index_test = np.where(y_test.reshape(-1) == 5)
    indexes = np.concatenate((class_0_index, class_1_index, class_2_index)).reshape(-1, 1)
    indexes_test = np.concatenate((class_0_index_test, class_1_index_test, class_2_index_test)).reshape(-1, 1)
    x_train_class_2 = X_train[indexes]
    y_train_class_2 = y_train[indexes].reshape(-1, 1)
    targets = {2: 0, 3: 1, 5: 2}
    x_train2 = X_train[indexes]
    y_train1 = y_train[indexes]
    x_test2 = X_test[indexes_test]
    y_test1 = y_test[indexes_test]
    y_train2 = np.copy(y_train1)
    y_test2 = np.copy(y_test1)
    for key, value in targets.items():  # Taking key and values from dictionary.
        y_train2[y_train1 == key] = value  # Matching the items where the item in array is same as in the dictionary. Setting it's value to the value of dictionary
        y_test2[y_test1 == key] = value
    x_train2 = x_train2[:, 0, :, :, :]# Corrigindo
    x_test2 = x_test2[:, 0, :, :, :]
    if resize > 0:
        data_scaled = np.zeros((x_train2.shape[0], resize, resize, 3))
        print(x_train2.shape, data_scaled.shape, x_test2.shape)
        for i, img in enumerate(x_train2):# Scaled to fit models
            large_img = cv2.resize(img, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            data_scaled[i] = large_img
        x_train2 = data_scaled
        data_scaled = np.zeros((x_test2.shape[0], resize, resize, 3))
        for i, img in enumerate(x_test2):
            large_img = cv2.resize(img, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            data_scaled[i] = large_img
        x_test2 = data_scaled
    y_train2 = y_train2[:, 0, :]# Corrigindo
    y_test2 = y_test2[:, 0, :]


    print(y_train.shape, y_train.shape)
    print(x_train2.shape, y_train2.shape,x_test2.shape)
    print(x_train_class_2.shape, y_train_class_2.shape)
    print(indexes.shape, indexes.shape)
    print(x_train2[1, 0, 0] == x_train_class_2[1,0, 0, 0])
    return x_train2, y_train2, x_test2, y_test2


#loadCifar(resize=64)