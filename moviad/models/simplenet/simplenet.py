"""PyTorch model for the SuperSimpleNet model implementation.

See Also:
    :class:`anomalib.models.image.supersimplenet.lightning_model.Supersimplenet`:
        SuperSimpleNet Lightning model.
"""

# Original Code
# Copyright (c) 2024 Blaž Rolih
# https://github.com/blaz-r/SuperSimpleNet.
# SPDX-License-Identifier: MIT
#
# Modified
# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from moviad.models.components.simplenet.discriminator import Discriminator
from moviad.models.components.simplenet.anomaly_generator import AnomalyGenerator
from moviad.models.components.simplenet.feature_extractor import UpscalingFeatureExtractor
from moviad.models.components.blur import GaussianBlur

class SimpleNetModel(nn.Module):
    """SuperSimpleNet Pytorch model.

    It consists of feature extractor, feature adaptor, anomaly generation mechanism and segmentation-detection module.

    Args:
        perlin_threshold (float): threshold value for Perlin noise thresholding during anomaly generation.
        backbone (str): backbone name
        layers (list[str]): backbone layers utilised
        stop_grad (bool): whether to stop gradient from class. to seg. head.
    """

    def __init__(
        self,
        feature_extractor,
        perlin_threshold: float = 0.2,
        stop_grad: bool = True,
    ) -> None:
        super().__init__()

        # feature extractor and pool
        self.feature_extractor = UpscalingFeatureExtractor(feature_extractor)
        channels = self.feature_extractor.get_channels_dim()

        # feature adapter
        self.adaptor = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
        )
        self.adaptor.apply(init_weights)

        self.segdec = Discriminator(channel_dim=channels, stop_grad=stop_grad)
        self.anomaly_generator = AnomalyGenerator(noise_mean=0, noise_std=0.015, threshold=perlin_threshold)

        self.anomaly_map_generator = AnomalyMapGenerator(sigma=4)

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        """SuperSimpleNet forward pass.

        Extract and process features, adapt them, generate anomalies (train only) and predict anomaly map and score.

        Args:
            images (torch.Tensor): Input images.
            masks (torch.Tensor): GT masks.
            labels (torch.Tensor): GT labels.

        Returns:
            inference: anomaly map and score
            training: anomaly map, score and GT masks and labels
        """
        output_size = images.shape[-2:]

        features = self.feature_extractor(images)
        adapted = self.adaptor(features)

        if self.training:
            masks = self.downsample_mask(masks, *features.shape[-2:])
            # make linter happy :)
            if labels is not None:
                labels = labels.type(torch.float32)

            features, masks, labels = self.anomaly_generator(
                adapted,
                masks,
                labels,
            )

            anomaly_map, anomaly_score = self.segdec(features)
            return anomaly_map, anomaly_score, masks, labels

        anomaly_map, anomaly_score = self.segdec(adapted)
        anomaly_map = self.anomaly_map_generator(anomaly_map, final_size=output_size)

        return anomaly_map, anomaly_score

    @staticmethod
    def downsample_mask(masks: torch.Tensor, feat_h: int, feat_w: int) -> torch.Tensor:
        """Downsample the masks according to the feature dimensions.

        Primarily used in supervised setting.

        Args:
            masks (torch.Tensor): input GT masks
            feat_h (int): feature height.
            feat_w (int): feature width.

        Returns:
            (torch.Tensor): downsampled masks.
        """
        masks = masks.type(torch.float32)
        # best downsampling proposed by DestSeg
        masks = F.interpolate(
            masks,
            size=(feat_h, feat_w),
            mode="bilinear",
        )
        return torch.where(
            masks < 0.5,
            torch.zeros_like(masks),
            torch.ones_like(masks),
        )


def init_weights(module: nn.Module) -> None:
    """Init weight of the model.

    Args:
        module (nn.Module): torch module.
    """
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)



class AnomalyMapGenerator(nn.Module):
    """Final anomaly map generator, responsible for upscaling and smoothing.

    Args:
        sigma (float) Gaussian kernel sigma value.
    """

    def __init__(self, sigma: float) -> None:
        super().__init__()
        kernel_size = 2 * math.ceil(3 * sigma) + 1
        self.smoothing = torchvision.transforms.GaussianBlur(kernel_size, sigma=0.4)

    def forward(self, out_map: torch.Tensor, final_size: tuple[int, int]) -> torch.Tensor:
        """Upscale and smooth anomaly map to get final anomaly map of same size as input image.

        Args:
            out_map (torch.Tensor): output anomaly map from seg. head.
            final_size (tuple[int, int]): size (h, w) of final anomaly map.

        Returns:
            torch.Tensor: final anomaly map.
        """
        # upscale & smooth
        anomaly_map = F.interpolate(out_map, size=final_size, mode="bilinear")
        return self.smoothing(anomaly_map)