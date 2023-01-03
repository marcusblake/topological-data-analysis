import torch
import torch.nn as nn
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn import SignatureLoss

"""
This topological autoencoder is written as described in https://arxiv.org/abs/1906.00722.
It is written to be trained on EMNIST data set.
"""
class TopologicalAutoencoderLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Computes persistent homology up to dimension 0 and uses L2 distance to calculate
        # distance matrix.
        self.rips_complex = VietorisRipsComplex(dim=0, p=2)
        self.top_loss = SignatureLoss(p=2)

    def _compute_topological_loss(self, x, z):
        x_persistence_infos = self.rips_complex(x)
        latent_persistence_infos = self.rips_complex(z)
        return self.top_loss([x, x_persistence_infos], [z, latent_persistence_infos])

    def forward(self, x: torch.tensor, z: torch.tensor) -> torch.tensor:
        """
        During forward pass of topological autoencoder, compute the persistent homology
        of both the data and the latent space. This returns the loss of the topological autoencoder,
        which is a linear combination of the reconstruction loss from the underlying base autoencoder and
        the addition of the topological loss, which is controlled by parameter lambda which indicates 
        the strength of topological regularization.

        Input: Batch of input data
        Output: Loss.
        """
        return self._compute_topological_loss(x.float(), z.float())