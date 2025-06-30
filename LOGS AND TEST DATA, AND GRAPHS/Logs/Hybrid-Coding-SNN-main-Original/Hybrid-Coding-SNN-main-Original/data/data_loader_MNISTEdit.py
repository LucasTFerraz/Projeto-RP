"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
Extracted from: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from data.autoaugment import CIFAR10Policy, Cutout
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


class SubLoader(MNIST):
    def __init__(self, *args, exclude_list=None, **kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        if exclude_list is None:
            exclude_list = []

        if self.train:

            labels = np.array(self.targets)
            class_l_index = []
            for x in range(10):
                class_T_index = np.where(labels.reshape(-1) == x)
                i = int(round(class_T_index[0].shape[0] / 10))
                class_l_index.append(class_T_index[0][:i])
            indexes = np.concatenate(class_l_index)
            self.data = self.data[indexes]
            self.targets = self.targets[indexes]
        else:
            labels = np.array(self.targets)
            class_l_index = []
            for x in range(10):
                class_T_index = np.where(labels.reshape(-1) == x)
                i = int(round(class_T_index[0].shape[0] / 10))
                class_l_index.append(class_T_index[0][:i])
            indexes = np.concatenate(class_l_index)
            self.data = self.data[indexes]
            self.targets = np.array(self.targets[indexes]).tolist()
            print(len(self.targets), self.data.shape)


def build_data(dpath: str, batch_size=128, cutout=False, workers=0, use_cifar10=False, auto_aug=False, image_s=32):
    image_size = (image_s, image_s)
    aug = [transforms.Grayscale(num_output_channels=3), transforms.RandomCrop(28, padding=4),
           transforms.RandomHorizontalFlip(), transforms.ToTensor()]

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    # aug.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
    aug.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    aug.append(transforms.Resize(image_size))
    transform_train = transforms.Compose(aug)
    transform_test = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                         transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], )

                                         # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), transforms.Resize(image_size),
                                         ])
    train_dataset = SubLoader(root=dpath, train=True, download=True, transform=transform_train, exclude_list=[1, 10])
    val_dataset = SubLoader(root=dpath, train=False, download=True, transform=transform_test, exclude_list=[1, 10])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, len(train_dataset.classes)
