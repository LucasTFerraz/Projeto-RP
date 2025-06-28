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
from torchvision.datasets import CIFAR10, CIFAR100
#from data.autoaugment import CIFAR10Policy, Cutout
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

class SubLoader(CIFAR10):
    def __init__(self, *args, exclude_list=None, **kwargs):
        super(SubLoader, self).__init__(*args, **kwargs)

        if exclude_list is None:
            exclude_list = []

        if self.train:

            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
            new_list = self.classes
            original_i = self.class_to_idx

            for i in sorted(exclude_list, reverse=True):
                del new_list[i]

            self.classes = new_list
            self.class_to_idx = {_class: i for i, _class in enumerate(new_list)}
            index_switch = {original_i[x]: y for x, y in self.class_to_idx.items()}
            self.data = np.delete(self.data, ~mask, axis=0)
            old_targets = np.delete(self.targets, ~mask, axis=0).tolist()
            self.targets = [index_switch[x] for x in old_targets]
            print(len(self.targets),self.data.shape)

        else:
            labels = np.array(self.targets)
            exclude = np.array(exclude_list).reshape(1, -1)
            mask = ~(labels.reshape(-1, 1) == exclude).any(axis=1)
            new_list = self.classes
            original_i = self.class_to_idx

            for i in sorted(exclude_list, reverse=True):
                del new_list[i]

            self.classes = new_list
            self.class_to_idx = {_class: i for i, _class in enumerate(new_list)}
            index_switch = {original_i[x]: y for x, y in self.class_to_idx.items()}
            self.data = np.delete(self.data, ~mask, axis=0)
            old_targets = np.delete(self.targets, ~mask, axis=0).tolist()
            self.targets = [index_switch[x] for x in old_targets]
            print(len(self.targets), self.data.shape)


def build_data(dpath: str, batch_size=128, cutout=False, workers=0, use_cifar10=True, auto_aug=False,image_s = 32):
    image_size = (image_s, image_s)
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        aug.append(
            transforms.Resize(image_size))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),transforms.Resize(image_size)
        ])

        train_dataset = SubLoader(root=dpath, train=True, download=True,transform=transform_train, exclude_list=[0, 1, 4, 6, 7, 8, 9])
        val_dataset = SubLoader(root=dpath, train=False, download=True,transform=transform_test, exclude_list=[0, 1, 4, 6, 7, 8, 9])

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=dpath, train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=dpath, train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)
    print(val_dataset.data.shape)
    return train_loader, val_loader, len(train_dataset.classes)
