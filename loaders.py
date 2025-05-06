from os.path import join, exists
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms
import torch


def load_SVHN():
    data_dir = 'Datasets/SVHN/'

    transform = transforms.ToTensor()  # No normalization, just 0–1 scaling

    train_set = torchvision.datasets.SVHN(
        root=data_dir,
        split='train',
        download=False,
        transform=transform
    )

    test_set = torchvision.datasets.SVHN(
        root=data_dir,
        split='test',
        download=False,
        transform=transform
    )

    X_train = torch.stack([x[0] for x in train_set])
    y_train = torch.tensor([x[1] for x in train_set], dtype=torch.long)
    y_train[y_train == 10] = 0

    X_test = torch.stack([x[0] for x in test_set])
    y_test = torch.tensor([x[1] for x in test_set], dtype=torch.long)
    y_test[y_test == 10] = 0

    return X_train, y_train, X_test, y_test



def load_CIFAR10():
    data_dir = 'Datasets/CIFAR-10/'

    transform = transforms.Compose([
        transforms.ToTensor()  # Converts PIL to float32 tensor in [0, 1]
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,  # Assumes it's already there
        transform=transform
    )

    test_set = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    X_train = torch.stack([x[0] for x in train_set])  # shape [N, C, H, W]
    y_train = torch.tensor([x[1] for x in train_set], dtype=torch.long)

    X_test = torch.stack([x[0] for x in test_set])
    y_test = torch.tensor([x[1] for x in test_set], dtype=torch.long)

    return X_train, y_train, X_test, y_test




def load_FASHION_MNIST():
    data_dir = 'Datasets/FASHION-MNIST/'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )

    test_set = torchvision.datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    X_train = torch.stack([x[0] for x in train_set])
    y_train = torch.tensor([x[1] for x in train_set], dtype=torch.long)

    X_test = torch.stack([x[0] for x in test_set])
    y_test = torch.tensor([x[1] for x in test_set], dtype=torch.long)

    return X_train, y_train, X_test, y_test


def load_MNIST():
    data_dir = 'Datasets/MNIST/'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=False,
        transform=transform
    )

    test_set = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=False,
        transform=transform
    )

    X_train = torch.stack([x[0] for x in train_set])
    y_train = torch.tensor([x[1] for x in train_set], dtype=torch.long)

    X_test = torch.stack([x[0] for x in test_set])
    y_test = torch.tensor([x[1] for x in test_set], dtype=torch.long)

    return X_train, y_train, X_test, y_test

