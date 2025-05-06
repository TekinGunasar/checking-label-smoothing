import torch

from torch import nn

from tqdm import tqdm

import torch.nn.functional as F

device = 'cpu'

one_hot_format = F.one_hot(torch.arange(10), num_classes=10).float()

# Label-smoothing training
def train_standard_model_delta(n_epochs, base_model, loader,delta, lr=1e-4, n_classes=10):
    model = base_model(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(n_epochs)):
        model.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            log_probs = F.log_softmax(logits,dim = 1)

            batch_size = y.size(0)
            soft_targets = torch.full(
                (batch_size, n_classes),
                fill_value=(1 - delta) / (n_classes - 1),
                device=device
            )

            soft_targets[torch.arange(batch_size), y] = delta

            optimizer.zero_grad()
            loss = -(soft_targets * log_probs).sum(dim = 1).mean()
            loss.backward()

            optimizer.step()

    return model



# One-hot (standard) training
def train_standard_model(n_epochs, base_model,loader,lr=1e-4, n_classes=10,loss_fn = None):
    model = base_model(n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(n_epochs)):
        model.train()
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)

            log_probs = F.log_softmax(logits,dim = 1)

            y_onehot = one_hot_format[y]

            optimizer.zero_grad()

            if loss_fn:
                loss = loss_fn(logits,y_onehot)
            else:
                loss = -(y_onehot * log_probs).sum(dim = 1).mean()


            loss.backward()
            optimizer.step()

    return model

# Evaluation
def get_test_acc(model,loader):
    model.eval()
    n_total, n_correct = 0, 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            n_correct += (preds == y).sum().item()
            n_total += y.size(0)

    test_acc = n_correct / n_total

    return test_acc

