import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from enum import Enum
import numpy as np
from collections import namedtuple
from loss import TopologicalAutoencoderLoss
import torchvision.transforms as transforms
from torchvision.datasets.mnist import EMNIST
from models import Autoencoder, ConvolutionalAutoencoder, VariationalAutoencoder


Hyperparameters = namedtuple('Hyperparameters', ['lr', 'top_reg', 'batch_size'])

class ModelType(Enum):
    autoencoder = 'autoencoder'
    top_autoencoder = 'top_autoencoder'
    vae = 'variational'
    conv_ae = 'convolutional'
    top_conv_ae = 'top_convolutional'

    def __str__(self):
        return self.value

# Constants.
_NUM_EPOCHS = 1 # Using same number of epochs as described in topological autoencoder paper.
_MAX_PIXEL_VALUE = 255.0

def _normalize_image(image: np.ndarray) -> np.ndarray:
    return image / _MAX_PIXEL_VALUE

EMNIST_IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(_normalize_image),
    transforms.Lambda(torch.flatten)
])

CONV_EMINIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(_normalize_image),
])

def create_nn_module_from_model_type(model_type: ModelType) -> nn.Module:
    if model_type == ModelType.autoencoder or model_type == ModelType.top_autoencoder:
        return Autoencoder(2)
    elif model_type == ModelType.conv_ae or model_type == ModelType.top_conv_ae:
        return ConvolutionalAutoencoder(2)
    else:
        return VariationalAutoencoder(2)

def train_model(model_type: ModelType, hyperparams: Hyperparameters, filepath: str) -> nn.Module:
    model = create_nn_module_from_model_type(model_type)
    optimizer = optim.SGD(model.parameters(), lr=hyperparams.lr)
    if model_type == ModelType.conv_ae or model_type == ModelType.top_conv_ae:
        transform = CONV_EMINIST_TRANSFORM
        print("conv")
    else:
        transform = EMNIST_IMAGE_TRANSFORM
    train_dataset = EMNIST(filepath, split='letters', train=True, download=True, transform=transform)
    data_loader = DataLoader(train_dataset, batch_size=hyperparams.batch_size, shuffle=True)
    n_samples = len(train_dataset)
    print(f'Training on dataset of size {n_samples}')
    print(model)
    mse = nn.MSELoss()
    topological_loss = TopologicalAutoencoderLoss()
    for epoch in range(_NUM_EPOCHS):
        epoch_loss = 0.0
        for batch_samples, _ in data_loader:
            forward_output = model(batch_samples.float())
            X_tilde, Z = forward_output.output, forward_output.latent_rep
            optimizer.zero_grad()
            if hyperparams.top_reg > 0.0:
                loss = mse(batch_samples.float(), X_tilde.float()) + \
                        hyperparams.top_reg * topological_loss(batch_samples.float(), Z.float())
            else:
                loss = mse(batch_samples.float(), X_tilde.float())
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        print('epoch [{}/{}], avg loss:{:.4f}'.format(epoch + 1, _NUM_EPOCHS + 1, epoch_loss / n_samples))
    print('Done training.')
    return model

def save_model(filepath: str, model: nn.Module):
    torch.save(model.state_dict(), filepath)

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--learning_rate', required=True, type=float, help='Learning rate for gradient descent.')
    arg_parser.add_argument('--momentum', type=float, help='[Optional] Momentum for gradient descent.')
    arg_parser.add_argument('--model', required=True, type=ModelType, choices=list(ModelType))
    arg_parser.add_argument('--topological_reg', type=float, default=0.0, help='[Optional] Topological regularization weight.')
    arg_parser.add_argument('--batch_size', required=True, type=int, help='Training batch size.')
    arg_parser.add_argument('--data_directory', required=True, help="Relative filepath to directory which contains dataset. If the dataset doesn't exist, this specifies the data where the dataset will be downloaded.")
    arg_parser.add_argument('--model_filename', type=str, help="Filename to use for saving the model.")

    args = arg_parser.parse_args()

    if args.model.value.startswith('top') and args.topological_reg == 0.0:
        raise ValueError("must specify topological regularization term when using a topological autoencoder.")

    hyperparams = Hyperparameters(args.learning_rate, args.topological_reg, args.batch_size)
    print(hyperparams)
    model = train_model(args.model, hyperparams, args.data_directory)
    if args.model_filename:
        save_model(args.model_filename, model)

if __name__ == '__main__':
    main()

