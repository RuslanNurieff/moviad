import unittest
import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode

from benchmark_common import mvtec_train_dataset, mvtec_test_dataset, real_iad_train_dataset, real_iad_test_dataset, \
    visa_train_dataset, visa_test_dataset
from moviad.entrypoints.patchcore import train_patchcore, PatchCoreArgs

backbones = {
    "mobilenet_v2": ["features.4", "features.7", "features.10"],
    # "wide_resnet50_2": ["layer1", "layer2", "layer3"],
    "phinet_1.2_0.5_6_downsampling": [2, 6, 7],
    "micronet-m1": [2, 4, 5],
    "mcunet-in3": [3, 6, 9],
    "resnet18": ["layer1", "layer2", "layer3"]
}


class PatchCoreBenchmarkContaminated(unittest.TestCase):
    def setUp(self):
        self.seed = 3
        self.epoch = 10
        torch.manual_seed(self.seed)
        self.args = PatchCoreArgs()
        self.args.contamination_ratio = 0.25
        self.args.batch_size = 1
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.img_input_size = (224, 224)

        self.transform = transforms.Compose([
            transforms.Resize(self.args.img_input_size),
            transforms.PILToTensor(),
            transforms.Resize(
                self.args.img_input_size,
                antialias=True,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ConvertImageDtype(torch.float32),
        ])

        wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))

    def test_patchcore_mvtec(self):
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
            self.logger = wandb.init(project="moviad_benchmark", group="patchcore")
            self.logger.config.update({
                "ad_model": "patchcore",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["patchcore", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"patchcore_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_patchcore(self.args, self.logger)
            self.logger.finish()

    def test_patchcore_realiad(self):
        self.args.train_dataset = real_iad_train_dataset
        self.args.test_dataset = real_iad_test_dataset
        self.args.train_dataset.class_name = "pcb"
        self.args.test_dataset.class_name = "pcb"
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
            self.logger = wandb.init(project="moviad_benchmark", group="patchcore")
            self.logger.config.update({
                "ad_model": "patchcore",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["patchcore", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"patchcore_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_patchcore(self.args, self.logger)
            self.logger.finish()

    def test_patchcore_visa(self):
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
            self.logger = wandb.init(project="moviad_benchmark", group="patchcore")
            self.logger.config.update({
                "ad_model": "patchcore",
                "dataset": type(self.args.train_dataset).__name__,
                "category": self.args.category,
                "backbone": self.args.backbone,
                "ad_layers": self.args.ad_layers,
                "seed": self.seed,
                "contamination_ratio": self.args.contamination_ratio,
                "contamination": self.contamination
            }, allow_val_change=True)
            self.logger.tags = ["patchcore", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"patchcore_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_patchcore(self.args, self.logger)
            self.logger.finish()


if __name__ == '__main__':
    unittest.main()
