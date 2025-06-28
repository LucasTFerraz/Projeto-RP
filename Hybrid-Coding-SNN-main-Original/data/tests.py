import os

import numpy as np
os.environ["KERAS_BACKEND"] = "torch"
from keras.datasets.cifar10 import load_data as cifar
def build_data(dpath: str, batch_size=128, cutout=False, workers=0, use_cifar10=False, auto_aug=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())


    aug.append(
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))

    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_dataset = CIFAR10(root=dpath, train=True, download=True, transform=transform_train)
    val_dataset = CIFAR10(root=dpath, train=False, download=True, transform=transform_test)
    return train_dataset, val_dataset, len(train_dataset.classes)
#[0, 1, 4, 6, 7, 8, 9]
(X_train, y_train), (X_test, y_test) = cifar()

class_0_index = np.where(y_train.reshape(-1) == 2)
class_1_index = np.where(y_train.reshape(-1) == 3)
class_index = []
for x in range(10):
    class_t_index = np.where(y_train.reshape(-1) == x)
    print(class_t_index.shape[0])
indexes = np.concatenate((class_0_index, class_1_index, class_2_index)).reshape(-1,1)
X_train_class_2 = X_train[indexes]
y_train_class_2 = y_train[indexes].reshape(-1,1)



X_train2 =  X_train[indexes]
y_train2 = y_train[indexes]
len(np.unique(y_train))
print(X_train.shape,y_train.shape)
print(X_train2.shape,y_train2.shape)