import profile
import unittest
from tokenize import group

import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode

from benchmark_common import mvtec_train_dataset, mvtec_test_dataset, real_iad_train_dataset, real_iad_test_dataset, \
    visa_train_dataset, visa_test_dataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum
from moviad.entrypoints.padim import train_padim, PadimArgs


backbones = {
    "mobilenet_v2": ["features.4", "features.7", "features.10"],
    # "wide_resnet50_2": ["layer1", "layer2", "layer3"],
    "phinet_1.2_0.5_6_downsampling": [2, 6, 7],
    "micronet-m1": [2, 4, 5],
    "mcunet-in3": [3, 6, 9],
}

class PadimBenchmark(unittest.TestCase):
    def setUp(self):
        self.seed = 3
        self.epoch = 10
        torch.manual_seed(self.seed)
        self.args = PadimArgs()
        self.args.contamination_ratio = 0.2
        self.args.batch_size = 2
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize(
                (256, 256),
                antialias=True,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ConvertImageDtype(torch.float32),
        ])
        wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))


    def test_padim_mvtec(self):
        self.args.train_dataset = mvtec_train_dataset
        self.args.test_dataset = mvtec_test_dataset
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.category = self.args.train_dataset.category
        self.contamination = 0
        if self.args.contamination_ratio > 0:
            self.args.train_dataset.contaminate(self.args.test_dataset, self.args.contamination_ratio)
            self.contamination = self.args.train_dataset.compute_contamination_ratio()
        for backbone, ad_layers in backbones.items():
            self.args.backbone = backbone
            self.args.ad_layers = ad_layers
            self.logger = wandb.init(project="moviad_benchmark", group="padim")
            self.logger.config.update({
                "ad_model": "padim",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["padim", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"padim_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_padim(self.args, self.logger)
            torch.cuda.empty_cache()
            self.logger.finish()

    def test_padim_realiad(self):
        self.args.train_dataset = real_iad_train_dataset
        self.args.test_dataset = real_iad_test_dataset
        self.args.train_dataset.class_name = RealIadClassEnum.PCB.value
        self.args.test_dataset.class_name = RealIadClassEnum.PCB.value
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.category = self.args.train_dataset.class_name
        self.contamination = 0
        if self.args.contamination_ratio > 0:
            self.args.train_dataset.contaminate(self.args.test_dataset, self.args.contamination_ratio)
            self.contamination = self.args.train_dataset.compute_contamination_ratio()
        for backbone, ad_layers in backbones.items():
            self.args.backbone = backbone
            self.args.ad_layers = ad_layers
            self.logger = wandb.init(project="moviad_benchmark", group="padim")
            self.logger.config.update({
                "ad_model": "padim",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["padim", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"padim_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_padim(self.args, self.logger)
            self.logger.finish()

    def test_padim_visa(self):
        self.args.train_dataset = visa_train_dataset
        self.args.test_dataset = visa_test_dataset
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.category = self.args.train_dataset.class_name
        self.contamination = 0
        if self.args.contamination_ratio > 0:
            self.args.train_dataset.contaminate(self.args.test_dataset, self.args.contamination_ratio)
            self.contamination = self.args.train_dataset.compute_contamination_ratio()
        for backbone, ad_layers in backbones.items():
            self.args.backbone = backbone
            self.args.ad_layers = ad_layers
            self.logger = wandb.init(project="moviad_benchmark", group="padim")
            self.logger.config.update({
                "ad_model": "padim",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["padim", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"padim_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_padim(self.args, self.logger)
            self.logger.finish()

if __name__ == '__main__':
    unittest.main()
