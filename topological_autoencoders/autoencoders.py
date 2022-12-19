import torch
import torch.nn as nn
from typing import override
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn import SignatureLoss


class AutoencoderBase(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.laten_dim = latent_dim
    def encode(self, x):
        pass
    def decode(self, x):
        pass
    def latent_space_dim(self):
        return self.laten_dim

"""
Traditional autoencoder architecture as implemented in https://www.cs.toronto.edu/~hinton/science.pdf.
Designed to be trained on EMNIST data set.
"""
class Autoencoder(nn.Module):
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

    

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z.float())

# """
# Convolutional autoencoder architecture inspired by http://users.cecs.anu.edu.au/~Tom.Gedeon/conf/ABCs2018/paper/ABCs2018_paper_58.pdf.
# """
# class ConvolutionalAutoencoder(AutoencoderBase):
#     def __init__(self, latent_space_dim: int = 2):
#         self.encoder =nn.Sequential(
#             nn.Conv2d(),
#             nn.ReLU(True),
#             nn.Conv2d()
#             nn
#         )
#         self.decoder = nn.Sequential(

#         )
#         self.latent_space_dim = latent_space_dim

#     def forward(self, x):
#         pass


"""
This topological autoencoder is written as described in https://arxiv.org/abs/1906.00722.
It is written to be trained on EMNIST data set.
"""
class TopologicalAutoencoder(AutoencoderBase):
    def __init__(self, base_autoencoder: AutoencoderBase, top_reg: int):
        super().__init__(base_autoencoder.latent_space_dim())
        self.base_autoencoder = base_autoencoder
        # Computes persistent homology up to dimension 0 and uses L2 distance to calculate
        # distance matrix.
        self.rips_complex = VietorisRipsComplex(dim=0, p=2)
        self.top_reg = top_reg
        self.top_loss = SignatureLoss(p=2)
        self.recon_loss = nn.MSELoss()

    def _compute_topological_loss(self, x, z):
        x_persistence_infos = self.rips_complex(x)
        latent_persistence_infos = self.rips_complex(z)
        return self.top_loss([x, x_persistence_infos], [z, latent_persistence_infos])

    def encode(self, x):
        return self.base_autoencoder.encoder(x)

    def decode(self, x):
        return self.base_autoencoder.decoder(x)

    def forward(self, x):
        """
        During forward pass of topological autoencoder, compute the persistent homology
        of both the data and the latent space. This returns the loss of the topological autoencoder,
        which is a linear combination of the reconstruction loss from the underlying base autoencoder and
        the addition of the topological loss, which is controlled by parameter lambda which indicates 
        the strength of topological regularization.

        Input: Batch of input data
        Output: Loss.
        """
        z = self.encode(x)
        topoological_loss = self._compute_topological_loss(x, z.float())
        output = self.decode(z)
        return self.recon_loss(output, x) + self.top_reg * topoological_loss
