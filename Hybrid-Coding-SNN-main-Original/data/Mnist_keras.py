import os

import cv2

os.environ["KERAS_BACKEND"] = "torch"
from keras.datasets.mnist import load_data as mnist
import keras.layers

import numpy as np


# [0, 1, 4, 6, 7, 8, 9]
def loadCifar(resize=0):
    # [0, 1, 4, 6, 7, 8, 9]
    (X_train, y_train), (X_test, y_test) = mnist()

    class_l_index = []
    class_l_index_test = []
    print(X_train.shape)
    for x in range(10):
        class_T_index = np.where(y_train.reshape(-1) == x)
        class_T_index_test = np.where(y_train.reshape(-1) == x)
        i = int(round(class_T_index[0].shape[0] / 10))
        class_l_index.append(class_T_index[0][:i])
        class_l_index_test.append(class_T_index_test[0][:i])
    indexes = np.concatenate(class_l_index)
    indexes_test = np.concatenate(class_l_index_test)
    x_train2 = X_train[indexes]
    y_train2 = y_train[indexes]
    x_test2 = X_test[indexes_test]
    y_test2 = y_test[indexes_test]
    if resize > 0:
        data_scaled = np.zeros((x_train2.shape[0], resize, resize,3))
        print(x_train2.shape, data_scaled.shape, x_test2.shape)
        for i, img in enumerate(x_train2):# Scaled to fit models
            large_img = cv2.resize(img, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            large_img2 = cv2.cvtColor(large_img, cv2.COLOR_GRAY2RGB)
            data_scaled[i] = large_img2
        x_train2 = data_scaled
        data_scaled = np.zeros((x_test2.shape[0], resize, resize,3))
        for i, img in enumerate(x_test2):
            large_img = cv2.resize(img, dsize=(resize, resize), interpolation=cv2.INTER_CUBIC)
            large_img2 = cv2.cvtColor(large_img, cv2.COLOR_GRAY2RGB)
            data_scaled[i] = large_img2
        x_test2 = data_scaled


    print(y_train.shape, y_train.shape)
    print(x_train2.shape, y_train2.shape,x_test2.shape)
    print(indexes.shape, indexes.shape)
    return x_train2, y_train2, x_test2, y_test2


#loadCifar(resize=64)