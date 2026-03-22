from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
from tqdm import tqdm

from moviad.models.training_args import TrainingArgs
from moviad.models.vad_model import VADModel

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

from collections import OrderedDict  # Add this import

class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super().__init__()

        out_planes = out_planes or in_planes
        layers = OrderedDict()  # Use OrderedDict instead of {}
        current_in_planes = in_planes

        for i in range(n_layers):
            layers[f"{i}_fc"] = torch.nn.Linear(current_in_planes, out_planes)

            if i < n_layers - 1 and layer_type > 1:
                layers[f"{i}_relu"] = torch.nn.LeakyReLU(0.2)

            current_in_planes = out_planes

        # Now Sequential will correctly unpack the named modules
        self.layers = torch.nn.Sequential(layers)
        self.apply(init_weight)

    def forward(self, x):
        return self.layers(x)

class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Linear(_in, _hidden),
                                     torch.nn.BatchNorm1d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Linear(_hidden, 1, bias=False)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x

@dataclass
class SimpleNetTrainArgs(TrainingArgs):
    disc_lr = 0.0002
    adp_lr = 1e-4
    meta_epochs = 40
    aed_meta_epochs = 1
    gan_epochs = 4
    mix_noise = 1
    noise_std = 0.05
    dsc_margin = 0.8
    weight_decay = 1e-5

    optimizer_disc = None
    optimizer_adp = None
    scheduler_disc = None

    def init_train(self, model: VADModel):
        if self.optimizer_disc is None:
            self.optimizer_disc = torch.optim.Adam(
                model.discriminator.parameters(),
                lr=self.disc_lr,
                weight_decay=self.weight_decay,
        )
        if self.optimizer_adp is None:
            self.optimizer_adp = torch.optim.AdamW(
                model.adaptor.parameters(),
                lr=self.adp_lr,
        )
        if self.scheduler_disc is None:
            self.scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_disc,
                (self.meta_epochs - self.aed_meta_epochs) * self.gan_epochs, self.disc_lr*.4
            )
    
    def __to_dict__(self):

        basic_dict = super().__to_dict__()

        return {
            **basic_dict,
            "disc_lr": self.disc_lr,
            "adp_lr": self.adp_lr,
            "meta_epochs": self.meta_epochs,
            "aed_meta_epochs": self.aed_meta_epochs,
            "gan_epochs": self.gan_epochs,
            "mix_noise": self.mix_noise,
            "noise_std": self.noise_std,
            "dsc_margin": self.dsc_margin,
            "weight_decay": self.weight_decay,
            "scheduler_disc": {
                "type": self.scheduler_disc.__class__.__name__,
                "T_max": self.scheduler_disc.T_max,
                "eta_min": self.scheduler_disc.eta_min,
            } if self.scheduler_disc else None,
        }



class SimpleNet(VADModel):

    def __init__(self,
                 feature_extractor,
                 target_embedding_dim=1536,
                 n_layers_proj=1,
                 n_layers_disc=2):

        super().__init__()
        self.feature_extractor = feature_extractor
        self.target_embedding_dim = target_embedding_dim
        self.adaptor = Projection(target_embedding_dim, target_embedding_dim, n_layers=n_layers_proj)
        self.discriminator = Discriminator(target_embedding_dim, n_layers=n_layers_disc)
        self.device = torch.device("cpu")

    def to(self, device: torch.device):
        super().to(device)
        self.device = device
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        self.adaptor.train(mode)
        self.discriminator.train(mode)
        self.feature_extractor.model.eval()
        return self

    def create_embeddings(self, batch):
        with torch.no_grad():
            features = self.feature_extractor(batch)

        output_shape = features[0].shape[2:]

        # align the spatial dimensions of all features to the same size
        for j in range(len(features)):
            features[j] = F.normalize(features[j], dim=1)
            features[j] = F.interpolate(features[j], size=output_shape, mode="bilinear", align_corners=False)

        # compress the feature dimensions of each level to a common dimension (preprocessing_dim)
        for j in range(len(features)):
            B, C, H, W = features[j].shape
            feat_flat = features[j].view(B, C, -1).transpose(1, 2).reshape(-1, 1, C)             # (B*H*W, 1, C)
            features[j] = F.adaptive_avg_pool1d(feat_flat, self.target_embedding_dim).squeeze(1) # (B*H*W, target_embedding_dim)

        features = torch.stack(features, dim=1)                                          # (B*H*W, num_layers, target_embedding_dim)
        features = features.view(features.shape[0], 1, -1)                               # (B*H*W, 1, num_layers*target_embedding_dim)
        features = F.adaptive_avg_pool1d(features, self.target_embedding_dim).squeeze(1) # (B*H*W, target_embedding_dim)

        return features, output_shape

    def train_step(self, batch: torch.Tensor, training_args: SimpleNetTrainArgs):
        adapted_features = self.forward(batch)

        noise_idxs = torch.randint(0, training_args.mix_noise, torch.Size([adapted_features.shape[0]]), device=self.device)              # (N,) for every sample in the batch, randomly select one of the K noise levels
        noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=training_args.mix_noise) # (N, K) one-hot encoding of the selected noise levels, for each sample in the batch, only one of the K noise levels is active (1), and the rest are inactive (0)
        noise = torch.stack(
            [torch.normal(0, training_args.noise_std * 1.1**(k), adapted_features.shape) for k in range(training_args.mix_noise)],
            dim=1
        ).to(self.device) # (N, K, C) for each sample in the batch, K different noise levels are generated, each with a different standard deviation (noise_std * 1.1^k), where k is the noise level index. The noise is sampled from a normal distribution with mean 0 and the specified standard deviation.
        noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1) # (N, C) for each sample in the batch, only the noise corresponding to the selected noise level is kept, and the rest are zeroed out
        fake_feats = adapted_features + noise

        scores = self.discriminator(torch.cat([adapted_features, fake_feats]))
        true_scores = scores[:len(adapted_features)]
        fake_scores = scores[len(adapted_features):]

        th = training_args.dsc_margin
        true_loss = torch.clip(-true_scores + th, min=0)
        fake_loss = torch.clip(fake_scores + th, min=0)

        loss = true_loss.mean() + fake_loss.mean()

        training_args.optimizer_disc.zero_grad()
        training_args.optimizer_adp.zero_grad()
        loss.backward()
        training_args.optimizer_disc.step()
        training_args.optimizer_adp.step()
        training_args.scheduler_disc.step()

        return loss.item()

    def train_epoch(self, epoch:int, train_dataloader: torch.utils.data.DataLoader, training_args: SimpleNetTrainArgs):

        avg_batch_loss = 0
        for e in range(training_args.gan_epochs):
            for batch in tqdm(train_dataloader):
                avg_batch_loss += self.train_step(batch, training_args)

        avg_batch_loss /= len(train_dataloader) * training_args.gan_epochs
        return avg_batch_loss

    def forward(self, batch):
        """Infer score and mask for a batch of images."""
        B, C, H, W = batch.shape

        sigma = 4
        blur = GaussianBlur(
            kernel_size=2 * int(4.0 * sigma + 0.5) + 1,
            sigma=sigma
        )

        batch = batch.to(self.device)
        features, feature_map_shape = self.create_embeddings(batch)
        adapted_features = self.adaptor(features)

        if self.training:
            return adapted_features
        else:
            patch_scores = -self.discriminator(adapted_features)

            # (B * H * W, 1) -> (B, 1, H, W)
            patch_scores = patch_scores.view(B, 1, feature_map_shape[0], feature_map_shape[1])

            image_scores = torch.amax(patch_scores, dim=(2, 3)) # (B, 1) max pooling over patch scores to get image-level scores

            # 6. Upsample to original image size
            # From (Batch, 1, H, W) -> (Batch, 1, target_size_H, target_size_W)
            upsampled_scores = F.interpolate(
                patch_scores, size=(H, W), mode="bilinear", align_corners=False
            )

            heatmaps = blur(upsampled_scores)

            return heatmaps, image_scores