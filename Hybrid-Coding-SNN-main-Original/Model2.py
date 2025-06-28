import numpy as np
import pandas as pd

import os
import cv2

os.environ["KERAS_BACKEND"] = "torch"
import torch

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications.vgg16 import VGG16
# from torchvision.models.vgg import vgg16,vgg16_bn
from keras.applications.inception_v3 import InceptionV3

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from keras.callbacks import Callback
from timeit import default_timer as timer


class TimingCallback(Callback):
    def __init__(self, logs={}):
        self.logs = []

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer() - self.starttime)


cb = TimingCallback()

# Load and preprocess the data
# data = "cifar"
# data ="brain"
# data = "poke"
# data = "dogs"
data = "mnist"
# classes = ['no', 'yes']

if "cifar" in data:
    from data.Cifar_keras import loadCifar

    batch_size = 32
    img_size = 75
    X_train, Y_train, X_val, Y_val = loadCifar(resize=img_size)
    classes = ["bird", "cat", "dog"]
    class_num = {"bird": 2, "cat": 3, "dog": 5}
    Y_train = to_categorical(Y_train, num_classes=len(classes))
    Y_val = to_categorical(Y_val, num_classes=len(classes))
elif "mnist" in data:
    from data.Mnist_keras import loadCifar

    batch_size = 32
    img_size = 75
    X_train, Y_train, X_val, Y_val = loadCifar(resize=img_size)
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_num = {x: x for x in classes}
    Y_train = to_categorical(Y_train, num_classes=len(classes))
    Y_val = to_categorical(Y_val, num_classes=len(classes))
else:
    if "dogs" in data:
        batch_size = 32
        img_size = 75
    elif "brain" in data:
        batch_size = 32
        img_size = 114
    elif "poke" in data:
        batch_size = 32
        img_size = 75
    else:
        batch_size = 32
        img_size = 75
    data_path = f'data/{data}/'
    classes = os.listdir(data_path)
    print(os.listdir(data_path))
    X = []
    Y = []
    for c in classes:
        path = os.path.join(data_path, c)
        class_num = classes.index(c)
        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            img_arr = cv2.resize(img_arr, (img_size, img_size))
            X.append(img_arr)
            Y.append(class_num)
            # print(img_arr.shape)

    X = np.array(X)
    Y = np.array(Y)
    Y = to_categorical(Y, num_classes=len(classes))
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
# Load the pre-trained VGG16 and InceptionV3 models

base_model1 = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model2 = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
# Split the data into training and validation sets

# Create the hybrid model
x1 = base_model1.output
x1 = GlobalAveragePooling2D()(x1)
x1 = Dense(1024, activation='relu')(x1)
x1 = Dropout(0.5)(x1)
predictions1 = Dense(len(classes), activation='softmax')(x1)

x2 = base_model2.output
x2 = GlobalAveragePooling2D()(x2)
x2 = Dense(1024, activation='relu')(x2)
x2 = Dropout(0.5)(x2)
predictions2 = Dense(len(classes), activation='softmax')(x2)

model = Model(inputs=[base_model1.input, base_model2.input], outputs=[predictions1, predictions2])
# Freeze the layers in the pre-trained models
for layer in base_model1.layers:
    layer.trainable = False

for layer in base_model2.layers:
    layer.trainable = False
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'accuracy'])

# Train the model

epochs = 12
# epochs = 30
print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape, )
history = model.fit([X_train, X_train], [Y_train, Y_train], batch_size=batch_size, epochs=epochs, verbose=1,
                    validation_data=([X_val, X_val], [Y_val, Y_val]), callbacks=[cb])
print(cb.logs)
print(sum(cb.logs))

# model.summary()

try:
    from keras.utils import plot_model

    plot_model(model, to_file='model.png', show_shapes=True)
except:
    pass

# Evaluate the model
score1 = model.evaluate([X_val, X_val], [Y_val, Y_val], verbose=0, return_dict=True)

score = model.evaluate([X_val, X_val], [Y_val, Y_val], verbose=0)
print(score1)
print(score)
print('Validation loss:', score1['loss'])
print('Validation accuracy:', score1['dense_3_accuracy'])

# Save the model
model.save(f'model{data}.h5')
