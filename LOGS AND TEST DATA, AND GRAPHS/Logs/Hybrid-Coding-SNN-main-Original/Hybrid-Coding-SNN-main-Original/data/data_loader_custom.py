"""
Create train, valid, test iterators for CIFAR-10 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
Extracted from: https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97
"""
import os
from collections import Counter

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from data.autoaugment import CIFAR10Policy, Cutout
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import InterpolationMode


def build_data(dpath: str, batch_size=128, cutout=False, workers=0, use_cifar10=False,
               auto_aug=False, image_s=32,):  # use_cifar10 não é usado, permaneceu só para facilitar
    padding = 4
    # padding = 4
    image_size = (image_s, image_s)
    # aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if "dogs" in dpath:
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_test = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:

        aug = [transforms.RandomCrop(image_s, padding=padding), transforms.RandomHorizontalFlip()]
        if auto_aug:
            aug.append(CIFAR10Policy())

        aug.append(transforms.ToTensor())

        if cutout:
            aug.append(Cutout(n_holes=1, length=16))
        aug.append(
            transforms.Resize(size=image_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
        )

        aug.append(
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), ])
        #dpath = os.path.join(dpath, 'data/Poke/')
        transform_train = transforms.Compose(aug)

    dataset = ImageFolder(root=dpath)

    lengh = len(dataset)
    size = int(round(0.8 * lengh))
    print(dataset.classes, '\n', dataset.class_to_idx)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [size, lengh - size])

    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_test
    print(len(train_dataset.indices))
    print(len(val_dataset.indices))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)
    return train_loader, val_loader, len(dataset.classes)
