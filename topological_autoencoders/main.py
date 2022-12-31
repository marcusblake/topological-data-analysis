import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from enum import Enum
from collections import namedtuple


Hyperparameters = namedtuple('Hyperparameters', ['lr'])

class ModelType(Enum):
    autoencoder = 'autoencoder'
    top_encoder = 'top_autoencoder'
    vae = 'variational'
    top_vae = 'top_variational'
    conv_ae = 'convolutional'
    top_conv_ae = 'top_convolutional'

    def __str__(self):
        return self.value

# Constants.
_NUM_EPOCHS = 200 # Using same number of epochs as described in topological autoencoder paper.


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


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--learning_rate', required=True, help='Learning rate for gradient descent.')
    arg_parser.add_argument('--momentum', help='[Optional] Momentum for gradient descent.')
    arg_parser.add_argument('--model', required=True, type=ModelType, choices=list(ModelType))

    arguments = arg_parser.parse_args()

if __name__ == '__main__':
    main()

