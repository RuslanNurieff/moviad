import unittest
import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode

from benchmark_common import mvtec_train_dataset, mvtec_test_dataset, backbones, real_iad_test_dataset, \
    real_iad_train_dataset, visa_train_dataset, visa_test_dataset
from main.common import INPUT_SIZES
from main_scripts.main_patchcore import IMAGE_SIZE
from moviad.entrypoints.stfpm import train_stfpm, STFPMArgs, test_stfpm

backbones = {
    "mobilenet_v2": [[8, 9], [10, 11, 12]],
    "wide_resnet50_2": [[3, 4, 5], [6, 7, 8]],
    "phinet_1.2_0.5_6_downsampling": [[2, 6, 7], [8, 9, 10]],
    "micronet-m1": [[2, 4, 5], [6, 8, 9]],
    "mcunet-in3": [[3, 6, 9], [12, 15, 18]],
}


class StfpmBenchmark(unittest.TestCase):
    def setUp(self):
        self.seed = 3
        self.epoch = 1
        torch.manual_seed(self.seed)
        self.args = STFPMArgs()
        self.args.contamination_ratio = 0.1
        self.args.batch_size = 4
        self.args.input_sizes = INPUT_SIZES
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.seeds = [self.seed]

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.Resize(
                IMAGE_SIZE,
                antialias=True,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ConvertImageDtype(torch.float32),
        ])

        wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))

    def test_stfpm_mvtec(self):
        self.args.train_dataset = mvtec_train_dataset
        self.args.test_dataset = mvtec_test_dataset
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.categories = [self.args.train_dataset.category]
        self.contamination = 0
        if self.args.contamination_ratio > 0:
            self.args.train_dataset.contaminate(self.args.test_dataset, self.args.contamination_ratio)
            self.contamination = self.args.train_dataset.compute_contamination_ratio()
        for backbone, ad_layers in backbones.items():
            self.args.backbone = backbone
            self.args.ad_layers = ad_layers
            self.logger = wandb.init(project="moviad_benchmark", group="stfpm")
            self.logger.config.update({
                "ad_model": "stfpm",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.categories,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["stfpm", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"stfpm_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_stfpm(self.args, self.logger)
            test_stfpm(self.args, self.logger)
            self.logger.finish()

    def test_stfpm_realiad(self):
        self.args.train_dataset = real_iad_train_dataset
        self.args.test_dataset = real_iad_test_dataset
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.categories = [self.args.train_dataset.class_name]
        self.contamination = 0
        if self.args.contamination_ratio > 0:
            self.args.train_dataset.contaminate(self.args.test_dataset, self.args.contamination_ratio)
            self.contamination = self.args.train_dataset.compute_contamination_ratio()
        for backbone, ad_layers in backbones.items():
            self.args.backbone = backbone
            self.args.ad_layers = ad_layers
            self.logger = wandb.init(project="moviad_benchmark", group="stfpm")
            self.logger.config.update({
                "ad_model": "stfpm",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.categories,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["stfpm", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"stfpm_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_stfpm(self.args, self.logger)
            self.logger.finish()

    def test_stfpm_visa(self):
        self.args.train_dataset = visa_train_dataset
        self.args.test_dataset = visa_test_dataset
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.categories = [self.args.train_dataset.class_name]
        self.contamination = 0
        if self.args.contamination_ratio > 0:
            self.args.train_dataset.contaminate(self.args.test_dataset, self.args.contamination_ratio)
            self.contamination = self.args.train_dataset.compute_contamination_ratio()
        for backbone, ad_layers in backbones.items():
            self.args.backbone = backbone
            self.args.ad_layers = ad_layers
            self.logger = wandb.init(project="moviad_benchmark", group="stfpm")
            self.logger.config.update({
                "ad_model": "stfpm",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.categories,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["stfpm", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"stfpm_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_stfpm(self.args, self.logger)
            self.logger.finish()

if __name__ == '__main__':
    unittest.main()
