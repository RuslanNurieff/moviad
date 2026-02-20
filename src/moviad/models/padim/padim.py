from __future__ import annotations
import os
from random import sample
from typing import Mapping, Union, Any, Dict, List, Tuple
from dataclasses import dataclass, field

import numpy as np
from tqdm import tqdm

from scipy.ndimage import gaussian_filter

import torch
from torch import nn
from torch.nn import functional as F

from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
from moviad.models.vad_model import VADModel
from moviad.models.training_args import TrainingArgs

# Dict: "backbone_model_name" -> {(layer_idxs): (true_dimension, random_projection_dimension)}
EMBEDDING_SIZES = {
    "phinet_1.2_0.5_6_downsampling": {
        (4, 5, 6): (200, 50),
        (5, 6, 7): (400, 100),
        (6, 7, 8): (576, 144),
        (2, 6, 7): (376, 94),
    },
    "micronet-m1": {
        (1, 2, 3): (40, 10),
        (2, 3, 4): (64, 16),
        (3, 4, 5): (112, 28),
        (2, 4, 5): (112, 28),
    },
    "mcunet-in3": {
        (3, 6, 9): (80, 20),
        (6, 9, 12): (112, 28),
        (9, 12, 15): (184, 46),
        (2, 6, 14): (136, 34),
    },
    "mobilenet_v2": {
        ("features.4", "features.7", "features.10"): (160, 40),
        ("features.7", "features.10", "features.13"): (224, 56),
        ("features.10", "features.13", "features.16"): (320, 80),
        ("features.3", "features.8", "features.14"): (248, 62),
    },
    "wide_resnet50_2": {("layer1", "layer2", "layer3"): (1792, 550)},
}


def idx_to_layer_name(backbone_model_name, idx: Union[Tuple, List]):
    if backbone_model_name in ["wide_resnet50_2"]:
        return tuple(f"layer{i}" for i in idx)
    elif backbone_model_name == "mobilenet_v2":
        return tuple(f"features.{i}" for i in idx)
    else:
        return idx


@dataclass
class PadimTrainArgs(TrainingArgs):
    """Training arguments for Padim.

    Padim does not use gradient-based optimization, so optimizer and
    loss_function are unused. Training consists of a single pass through
    the data to collect embeddings and fit a multivariate Gaussian.
    """
    diag_cov: bool = False

    def init_train(self, model: VADModel):
        # Padim has no optimizer or loss — training is statistical fitting
        pass


class Padim(VADModel):
    HYPERPARAMS = [
        "class_name",
        "backbone_model_name",
        "t_d",
        "d",
        "gauss_mean",
        "gauss_cov",
        "diag_cov",
        "layers_idxs",
    ]

    def __init__(
        self,
        backbone_model_name: str,
        layers_idxs: Union[Tuple, List],
        device: str,
        class_name: str = "",
        diag_cov: bool = False,
    ):
        """
        Args:
            backbone_model_name: one of the following strings: 'wide_resnet50_2', 'mobilenet_v2',
                'phinet_1.2_0.5_6_downsampling', 'micronet-m1', 'mcunet-in3'
            layers_idxs: indices of the layers to extract features from
            class_name: category name (e.g. 'bottle', 'cable', ...)
            diag_cov: if True, keep only the diagonal elements of the covariance matrices
        """
        super().__init__()
        self.diagonal_gauss_cov = None
        self.class_name = class_name
        self.device = device
        self.diag_cov = diag_cov
        # feature extractor backbone model
        self.backbone_model_name = backbone_model_name
        self.layers_idxs = layers_idxs

        self.layers_idxs = idx_to_layer_name(
            backbone_model_name, layers_idxs
        )  # feature extraction layers

        self.load_backbone()
        # dimensionality reduction: random projection
        random_dims = torch.tensor(sample(range(0, self.t_d), self.d))
        self.random_dimensions = torch.nn.Parameter(random_dims, requires_grad=False)
        # training: learn the multivariate Gaussian distribution from the extracted features
        self.train_outputs = None  # list of mean and covariance matrix numpy arrays
        self.gauss_mean = None
        self.gauss_cov = None

    @staticmethod
    def embedding_concat(x, y):
        B, C1, H1, W1 = x.size()
        _, C2, H2, W2 = y.size()
        s = int(H1 / H2)
        x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
        x = x.view(B, C1, -1, H2, W2)
        z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
        for i in range(x.size(2)):
            z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
        z = z.view(B, -1, H2 * W2)
        z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

        return z

    def raw_feature_maps_to_embeddings(
        self, layer_outputs: Dict[str, List[torch.Tensor]]
    ):
        """
        Given a dict of lists of outputs of the layers, concatenate the feature maps and
        eventually reduce the dimensionality to return the embedding vectors.

        - embedding vector shape: (B, C, H, W)
        - B = number of samples in the train set
        - C = number of "channels", or number of feature maps --> may be reduced by dim. reduction
        - H, W = height and width of the feature maps
        """
        # concatenate the outputs of the different dataloader batches
        output_tensors: dict[str, torch.Tensor] = {
            layer: torch.cat(outputs, 0) for layer, outputs in layer_outputs.items()
        }
        # concatenate the feature maps to get the raw embedding vectors
        embedding_vectors: torch.Tensor = output_tensors[self.layers_idxs[0]]
        for layer in self.layers_idxs[1:]:
            embedding_vectors = Padim.embedding_concat(
                embedding_vectors, output_tensors[layer]
            )
        # dimensionality reduction: select the random dimensions to reduce the embedding vectors
        embedding_vectors = torch.index_select(
            embedding_vectors.to(self.device), 1, self.random_dimensions
        )
        return embedding_vectors

    def forward(self, x):
        # 1. extract feature maps and get the raw layer outputs (conv. feature maps)
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.layers_idxs
        }
        # forward through the net to get the intermediate outputs with the hooks
        with torch.no_grad():
            # _ = self.backbone(x)
            _ = self.backbone(x)
        # get intermediate layer outputs
        for layer, output in zip(self.layers_idxs, self.outputs):  # new
            layer_outputs[layer].append(output.cpu().detach())  # new
        # initialize hook outputs
        self.outputs = []

        if self.training:
            return layer_outputs

        # ---- EVAL INFERENCE ----
        # 2. use the feature maps to get the embeddings
        embedding_vectors = self.raw_feature_maps_to_embeddings(layer_outputs)

        # 3. compute the distance matrix
        if self.diag_cov:
            dist_list = self.compute_distances_diagonal(embedding_vectors)
        else:
            dist_list = self.compute_distances(embedding_vectors)
        # 4. upsample
        score_map = (
            F.interpolate(
                dist_list.unsqueeze(1),
                size=x.size(2),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .numpy()
        )
        # 5. apply gaussian smoothing on the score map
        for i in range(score_map.shape[0]):
            score_map[i] = gaussian_filter(score_map[i], sigma=4)
        # 6. the image anomaly score is the maximum score in the score map
        img_scores = score_map.reshape(score_map.shape[0], -1).max(axis=1)

        # need to unsqueeze to have (batch, 1, H, W)
        score_map = np.expand_dims(score_map, axis=1)

        return torch.from_numpy(score_map), torch.from_numpy(img_scores)

    def to(self, device: torch.device):
        super().to(device)
        self.backbone_model.to(device)
        self.device = device
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # backbone is always frozen/eval for Padim
        self.backbone_model.model.eval()
        return self

    def parameters(self):
        # Padim has no trainable parameters (statistical fitting only)
        return iter([])

    def train_epoch(
        self, epoch, train_dataloader, training_args: PadimTrainArgs
    ):
        """Single-epoch training for Padim.

        Collects feature maps from all batches, computes embeddings,
        and fits the multivariate Gaussian distribution.

        Returns:
            float: 0.0 (Padim has no loss in the traditional sense)
        """
        # 1. collect feature maps from all batches
        layer_outputs: dict[str, list[torch.Tensor]] = {
            layer: [] for layer in self.layers_idxs
        }
        for batch in tqdm(train_dataloader, desc="| feature extraction | train |"):
            batch_outputs = self.train_step(batch, training_args)
            for layer, output in batch_outputs.items():
                layer_outputs[layer].extend(output)

        # 2. convert feature maps to embeddings
        embedding_vectors = self.raw_feature_maps_to_embeddings(layer_outputs)

        # 3. fit the multivariate Gaussian distribution
        diag_cov = getattr(training_args, 'diag_cov', self.diag_cov)
        if diag_cov:
            self.fit_multivariate_diagonal_gaussian(
                embedding_vectors, update_params=True
            )
        else:
            self.fit_multivariate_gaussian(
                embedding_vectors, update_params=True
            )

        return 0.0  # no loss for Padim

    def train_step(self, batch: torch.Tensor, training_args: TrainingArgs):
        """Single batch training step — extracts feature maps.

        Args:
            batch: Input image batch.
            training_args: Training arguments (unused for Padim).

        Returns:
            dict: Layer outputs (feature maps) for this batch.
        """
        batch = batch.to(self.device)
        return self.forward(batch)

    def fit_multivariate_diagonal_gaussian(
        self, embedding_vectors: torch.Tensor, update_params: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fit a multivariate Gaussian distribution to the set of given embedding vectors.

        Returns:
            List of mean and covariance matrix diagonal numpy arrays
        """
        B, C, H, W = embedding_vectors.size()

        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors.cpu(), dim=0).numpy()
        diagonal_cov = torch.zeros(C, H * W).numpy()
        I = np.identity(C)
        # for every "patch" in the feature map, compute the covariance across the batch
        for i in range(H * W):
            # TODO: use np.var instead of np.cov in diagonal covariance computation
            temp_cov = (
                np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False)
                + 0.01 * I
            )

            diagonal_cov[:, i] = np.diag(temp_cov)

        if update_params:
            self.gauss_mean, self.diagonal_gauss_cov = mean, diagonal_cov
        return mean, diagonal_cov

    def fit_multivariate_gaussian(self, embedding_vectors, update_params, logger=None):
        """
        Fit a multivariate Gaussian distribution to the set of given embedding vectors.

        Returns:
            List of mean and covariance matrix numpy arrays
        """
        B, C, H, W = embedding_vectors.size()

        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors.cpu(), dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        I = np.identity(C)
        # for every "patch" in the feature map, compute the covariance across the batch
        for i in range(H * W):
            if self.diag_cov:
                temp_cov = (
                    np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False)
                    + 0.01 * I
                )
                temp_cov[~I.astype(bool)] = 0
                cov[:, :, i] = temp_cov
            else:
                cov[:, :, i] = (
                    np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False)
                    + 0.01 * I
                )
            if logger is not None:
                logger.log(
                    {
                        "cov": cov[:, :, i],
                        "mean": mean[:, i],
                    }
                )
        if update_params:
            self.gauss_mean, self.gauss_cov = mean, cov
        return mean, cov

    def load_backbone(self):
        """
        Load the backbone model

        Args:
            backbone_model_name: one of the following strings: 'wide_resnet50_2', 'mobilenet_v2'
        """
        backbone = CustomFeatureExtractor(
            model_name=self.backbone_model_name,
            layers_idx=self.layers_idxs,
            device=self.device,
            frozen=True,
        )
        self.backbone_model = backbone

        # define the backbone behavior
        def backbone_forward(x):
            self.outputs = backbone(x)

        self.backbone = backbone_forward

        # save the true and random projection dimensions
        self.t_d, self.d = EMBEDDING_SIZES[self.backbone_model_name][
            tuple(self.layers_idxs)
        ]

    def get_model_savepath(self, save_path):
        return os.path.join(
            save_path,
            "checkpoints_%s" % self.backbone_model_name,
            "train_%s.pth.tar" % self.class_name,
        )

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # add all the hyperparameters to the state dict
        for p in self.HYPERPARAMS:
            state_dict[p] = getattr(self, p)
        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # load the hyperparameters
        for p in self.HYPERPARAMS:
            setattr(self, p, state_dict[p])
        # load the backbone models
        self.load_backbone()
        # remove the hyperparameters from the state dict
        state_dict = {k: v for k, v in state_dict.items() if k not in self.HYPERPARAMS}
        return super().load_state_dict(state_dict, strict=strict)

    def compute_distances(self, embedding_vectors: torch.Tensor):
        """
        Compute the Mahalanobis distances between the embedding vectors and the
        multivariate Gaussian distribution.
        """
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()
        assert (
            self.gauss_mean is not None and self.gauss_cov is not None
        ), "The model must be trained first."

        # (C, C, H*W) -> (H*W, C, C) -> inv -> (H*W, C, C)
        cov = np.transpose(self.gauss_cov, (2, 0, 1))  # (H*W, C, C)
        cov_inv = np.linalg.inv(cov)  # single batched call

        # (B, C, H*W) - (1, C, H*W) -> (B, C, H*W)
        diff = embedding_vectors - self.gauss_mean[np.newaxis, :, :]

        # Rearrange to (H*W, B, C) for batched matmul
        diff_t = np.transpose(diff, (2, 0, 1))  # (H*W, B, C)

        # Batched matmul: (H*W, B, C) @ (H*W, C, C) -> (H*W, B, C)
        temp = np.matmul(diff_t, cov_inv)

        # Mahalanobis distance squared: element-wise multiply and sum over C
        dist_sq = np.sum(temp * diff_t, axis=2)  # (H*W, B)

        dist_list = np.sqrt(dist_sq).T.reshape(B, H, W)  # (B, H, W)
        return torch.tensor(dist_list)

    def compute_distances_diagonal(self, embedding_vectors: torch.Tensor):
        """
        Compute the Mahalanobis distances between the embedding vectors and the
        multivariate Gaussian distribution (diagonal covariance).
        """
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W).cpu().numpy()
        assert (
            self.gauss_mean is not None and self.diagonal_gauss_cov is not None
        ), "The model must be trained first."

        # (B, C, H*W) - (1, C, H*W) -> (B, C, H*W)
        diff = embedding_vectors - self.gauss_mean[np.newaxis, :, :]

        # Mahalanobis with diagonal cov: sqrt(sum((x-mu)^2 / diag_cov))
        # diagonal_gauss_cov shape: (C, H*W)
        dist_sq = np.sum(diff ** 2 / self.diagonal_gauss_cov[np.newaxis, :, :], axis=1)  # (B, H*W)

        dist = np.sqrt(dist_sq).reshape(B, H, W)
        return torch.tensor(dist)
