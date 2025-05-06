from os.path import join, exists
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

SVHN_data_root_dir = 'Datasets/SVHN/'


def load_SVHN(split='train'):
    valid_splits = {'train', 'test', 'extra'}
    if split not in valid_splits:
        raise ValueError(f"Invalid split '{split}'. Must be one of {valid_splits}.")

    data_path = join(SVHN_data_root_dir, f'{split}_32x32.mat')
    if not exists(data_path):
        raise FileNotFoundError(f"SVHN .mat file not found at: {data_path}")

    data_mat = loadmat(data_path)

    if 'X' not in data_mat or 'y' not in data_mat:
        raise ValueError(f"Missing keys in .mat file: expected keys 'X' and 'y' in {data_path}")

    X, y = data_mat['X'], data_mat['y']
    X = np.transpose(X, (3, 2, 0, 1)).astype(np.float32) / 255.  # (N, C, H, W)
    y[y == 10] = 0

    X = torch.tensor(X)
    y = torch.tensor(y.squeeze(), dtype=torch.long)

    return X, y