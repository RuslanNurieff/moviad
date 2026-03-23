"""PadimLite: Lightweight PaDiM variant using diagonal covariance matrices.

Instead of storing and inverting full C×C covariance matrices per spatial
location, PadimLite keeps only the diagonal (per-channel variances).
This drastically reduces memory and compute at the cost of ignoring
cross-channel correlations.
"""

from __future__ import annotations
from typing import Union, Tuple, List

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from moviad.models.padim.padim import Padim, PadimTrainArgs


class PadimLite(Padim):
    """PaDiM with diagonal (isotropic per-channel) covariance."""

    HYPERPARAMS = [
        "class_name",
        "backbone_model_name",
        "t_d",
        "d",
        "gauss_mean",
        "diagonal_gauss_cov",
        "layers_idxs",
    ]

    def __init__(
        self,
        backbone_model_name: str,
        layers_idxs: Union[Tuple, List],
        device: str,
        class_name: str = "",
    ):
        super().__init__(backbone_model_name, layers_idxs, device, class_name)
        self.diagonal_gauss_cov = None  # (C, H*W)

    def train_epoch(self, epoch, train_dataloader, training_args: PadimTrainArgs):
        """Collect embeddings and fit diagonal Gaussian."""
        layer_outputs = {layer: [] for layer in self.layers_idxs}
        from tqdm import tqdm
        for batch in tqdm(train_dataloader, desc="| feature extraction | train |"):
            batch_outputs = self.train_step(batch, training_args)
            for layer, output in batch_outputs.items():
                layer_outputs[layer].extend(output)

        embedding_vectors = self.raw_feature_maps_to_embeddings(layer_outputs)
        self.fit_diagonal_gaussian(embedding_vectors)
        return 0.0

    def fit_diagonal_gaussian(self, embedding_vectors: torch.Tensor):
        """Fit mean and diagonal covariance from embedding vectors.

        Args:
            embedding_vectors: (B, C, H, W) tensor
        """
        B, C, H, W = embedding_vectors.size()
        X = embedding_vectors.view(B, C, H * W).cpu().numpy()  # (B, C, H*W)

        mean = X.mean(axis=0)  # (C, H*W)
        # Per-channel variance + regularization
        var = X.var(axis=0, ddof=1)  # (C, H*W), unbiased
        var += 0.01

        self.gauss_mean = mean
        self.diagonal_gauss_cov = var

    def compute_distances(self, embedding_vectors: torch.Tensor):
        """Mahalanobis distance with diagonal covariance."""
        B, C, H, W = embedding_vectors.size()
        X = embedding_vectors.view(B, C, H * W).cpu().numpy()
        assert (
            self.gauss_mean is not None and self.diagonal_gauss_cov is not None
        ), "The model must be trained first."

        diff = X - self.gauss_mean[np.newaxis, :, :]  # (B, C, H*W)
        dist_sq = np.sum(diff ** 2 / self.diagonal_gauss_cov[np.newaxis, :, :], axis=1)  # (B, H*W)
        dist = np.sqrt(dist_sq).reshape(B, H, W)
        return torch.tensor(dist)
