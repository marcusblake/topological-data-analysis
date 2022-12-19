import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def train_model(model: nn.Module, data_loader: DataLoader, learning_rate: float, epochs: int):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    n = len(data_loader)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_samples, _ in data_loader:
            output = model(batch_samples.float())
            optimizer.zero_grad()
            loss = criterion(batch_samples.float(), output.float())
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print('epoch [{}/{}], avg loss:{:.4f}'.format(epoch + 1, epochs, total_loss / n))
    print('Done training.')


def train_topological_model(model: nn.Module, data_loader: DataLoader, learning_rate: float, epochs: int):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    n = len(data_loader)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_samples, _ in data_loader:
            loss = model(batch_samples.float())
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print('epoch [{}/{}], avg loss:{:.4f}'.format(epoch + 1, epochs, total_loss / n))
    print('Done training.')

