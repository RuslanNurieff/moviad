import copy

import torch
from torchvision.transforms import GaussianBlur
import torch.nn.functional as F
import numpy as np

from models.R4AD.resnet import resnet18
from models.R4AD.de_resnet import de_resnet18

class RD4AD(torch.nn.Module):

    def __init__(self, model_name, device, input_size = (224, 224)):
        super().__init__()

        self.model_name = model_name
        self.device = device
        self.input_size = input_size

        self.encoder, self.bn = resnet18(pretrained=True)
        self.decoder = de_resnet18(pretrained=False)

    def to(self, device: torch.device):
        self.encoder.to(device)
        self.bn.to(device)
        self.decoder.to(device)

    def train(self, *args, **kwargs):
        self.encoder.eval()
        self.bn.train()
        self.decoder.train()
        return super().train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        self.encoder.eval()
        self.bn.eval()
        self.decoder.eval()
        return super().eval(*args, **kwargs)

    def forward(self, batch: torch.Tensor):
        """
        Output tensors
        List[torch.Tensor] of len (n_layers)
        every tensor shape is (B C H W)
        """
        enc_batch = self.encoder(batch)
        bn_batch = self.bn(enc_batch)
        dec_batch = self.decoder(bn_batch)

        if self.training:
            return enc_batch, bn_batch, dec_batch
        else:
            return self.post_process(enc_batch, dec_batch)

    def __call__(self, batch):
        return self.forward(batch)
