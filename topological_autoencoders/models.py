import torch
import torch.nn as nn
from collections import namedtuple

ForwardOutput = namedtuple('ForwardOutput', ['output', 'latent_rep'])

class AutoencoderBase(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.laten_dim = latent_dim
    def encode(self, x):
        pass
    def decode(self, x):
        pass

    def forward(self, x: torch.tensor) -> ForwardOutput:
        pass

    def latent_space_dim(self):
        return self.laten_dim

"""
Traditional autoencoder architecture as implemented in https://www.cs.toronto.edu/~hinton/science.pdf.
Designed to be trained on EMNIST data set.
"""
class Autoencoder(AutoencoderBase):
    def __init__(self, latent_space_dim):
        super().__init__(latent_space_dim)
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), # Size of each EMNIST image is 28 x 28.
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, latent_space_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28)
        )

    def encode(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)

    def decode(self, z: torch.tensor) -> torch.tensor:
        return self.decoder(z)

    def forward(self, x: torch.tensor) -> ForwardOutput:
        z = self.encoder(x)
        return ForwardOutput(self.decoder(z.float()), z.float())


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.tensor) -> torch.tensor:
        return x.view(*self.shape)

class ConvolutionalAutoencoder(AutoencoderBase):
    def __init__(self, latent_space_dim: int = 2):
        super().__init__(latent_space_dim)
        self.encoder =nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), # Maintains input dimensions 28x28. Include padding at input layer.
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsamples to 14x14
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1), # Maintins input dimension 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsamples to 7x7
            nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1), # Maintains dimension 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Downsamples to 3x3
            nn.Flatten(), # Flattens to vector of 3x3x8 since the number of filters from last conv2d layer is 8
            nn.Linear(8*3*3, latent_space_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_dim, 8*3*3),
            View((-1, 8, 3, 3)), # Reshapes to have 8 channels, and 3x3 arrays.
            nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2), # Upsamples to 7x7
            nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2), # Upsamples to 14x14
            nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2) # Upsamples to 28x28
        )
        self.latent_dim = latent_space_dim

    def encode(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)

    def decode(self, z: torch.tensor) -> torch.tensor:
        return self.decoder(z)

    def forward(self, x: torch.tensor) -> ForwardOutput:
        z = self.encode(x)
        return ForwardOutput(self.decode(z), z)

"""Implementation of variational autoencoder as described in https://arxiv.org/pdf/1312.6114.pdf"""
class VariationalAutoencoder(AutoencoderBase):
    def __init__(self, latent_space_dim: int = 2):
        super().__init__(latent_space_dim)
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128), # Size of each EMNIST image is 28 x 28.
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, latent_space_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_space_dim, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 28*28)
        )

    def encode(self, x: torch.tensor) -> torch.tensor:
        return self.encoder(x)

    def decode(self, z: torch.tensor) -> torch.tensor:
        return self.decoder(z)

    def forward(self, x: torch.tensor) -> ForwardOutput:
        z = self.encode(x)
        return ForwardOutput(self.decode(z), z)
