import unittest
from tokenize import group

import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode
from transformers.pipelines.question_answering import Dataset

from benchmark_common import mvtec_train_dataset, real_iad_train_dataset, visa_train_dataset, mvtec_test_dataset, \
    real_iad_test_dataset, visa_test_dataset, backbones
from moviad.datasets.builder import DatasetType, DatasetConfig
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum
from moviad.entrypoints.cfa import train_cfa, CFAArguments, test_cfa

class CfaBenchmark(unittest.TestCase):
    def setUp(self):
        self.seed = 4
        self.epoch = 1
        torch.manual_seed(self.seed)
        self.args = CFAArguments()
        self.args.contamination_ratio = 0.2
        self.args.batch_size = 4
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.config = DatasetConfig('./config.json')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.PILToTensor(),
            transforms.Resize(
                (224, 224),
                antialias=True,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ConvertImageDtype(torch.float32),
        ])
        wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))


    def test_cfa_mvtec(self):
        self.args.category = 'pill'
        self.args.dataset_type = DatasetType.MVTec
        self.args.dataset_config = self.config
        self.contamination = 0
        for backbone, ad_layers in backbones.items():
            self.args.backbone = backbone
            self.args.ad_layers = ad_layers
            self.logger = wandb.init(project="moviad_benchmark", group="cfa")
            self.logger.config.update({
                "ad_model": "cfa",
                "dataset": DatasetType.MVTec,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["cfa", self.args.dataset_type, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"cfa_{self.args.dataset_type}_{self.args.backbone}"
            train_cfa(self.args, self.logger)
            self.logger.finish()

    def test_cfa_realiad(self):
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
            self.logger = wandb.init(project="moviad_benchmark", group="cfa")
            self.args.backbone = backbone
            self.args.ad_layers = ad_layers
            self.logger.config.update({
                "ad_model": "cfa",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["cfa", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"cfa_{type(self.args.train_dataset).__name__}_{self.args.backbone}"

            train_cfa(self.args, self.logger)
            self.logger.finish()

    def test_cfa_visa(self):
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
            self.logger = wandb.init(project="moviad_benchmark", group="cfa")
            self.logger.config.update({
                "ad_model": "cfa",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["cfa", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"cfa_{type(self.args.train_dataset).__name__}_{self.args.backbone}"

            train_cfa(self.args, self.logger)
            self.logger.finish()

if __name__ == '__main__':
    unittest.main()
