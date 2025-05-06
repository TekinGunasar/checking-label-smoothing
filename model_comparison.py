import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch import nn

from loaders import load_SVHN,load_CIFAR10,load_FASHION_MNIST

from utils import *

import numpy as np
import random
import os

#def set_seed(random_seed):
#    os.environ['PYTHONHASHSEED'] = str(random_seed)
#    torch.manual_seed(random_seed)
#    torch.cuda.manual_seed(random_seed)
#    torch.cuda.manual_seed_all(random_seed)
#    np.random.seed(random_seed)
#    random.seed(random_seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(3072, 1024)
        self.dropout = nn.Dropout(0.3)
        self.output = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.hidden(x))
        x = self.dropout(x)
        return self.output(x)

X_train, y_train, X_test, y_test = load_SVHN()


train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

n_epochs = 10


delta = 0.2
delta_model = train_standard_model_delta(
    n_epochs = n_epochs,
    base_model = MLP,
    loader = train_loader,
    delta = delta
)

standard_model = train_standard_model(
    n_epochs = n_epochs,
    base_model = MLP,
    loader = train_loader
)


standard_model_direct_label_smoothing = train_standard_model(
    n_epochs = n_epochs,
    base_model = MLP,
    loader = train_loader,
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing = 1-delta)
)

#i.e placing 20% confidence in the true class when constructing soft targets


delta_test_acc = get_test_acc(
    delta_model,test_loader
)

standard_test_acc = get_test_acc(
    standard_model,test_loader
)

standard_test_acc_label_smoothing = get_test_acc(
    standard_model_direct_label_smoothing,test_loader
)

print(f'Delta Test Accuracy : {delta_test_acc}')

print(f'Standard Test Accuracy : {standard_test_acc}')

print(f'Standard with Label Smoothing : {standard_test_acc_label_smoothing}')
