import unittest
from tokenize import group

import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode

from benchmark_common import mvtec_train_dataset, mvtec_test_dataset, real_iad_train_dataset, real_iad_test_dataset
from datasets.visadataset_tests import VISA_DATASET_PATH, VISA_DATASET_CSV_PATH
from main_scripts.main_padim import main_train_padim
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.entrypoints.padim import train_padim, PadimArgs
from moviad.entrypoints.patchcore import IMAGE_SIZE
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_tests import REAL_IAD_DATASET_PATH, AUDIO_JACK_DATASET_JSON
from tests.main.common import TrainingArguments, get_training_args, MVTECH_DATASET_PATH


backbones = {
    "mobilenet_v2": ["features.4", "features.7", "features.10"],
    # "wide_resnet50_2": ["layer1", "layer2", "layer3"],
    "phinet_1.2_0.5_6_downsampling": [2, 6, 7],
    "micronet-m1": [2, 4, 5],
    "mcunet-in3": [3, 6, 9],
}

class PadimBenchmark(unittest.TestCase):
    def setUp(self):
        self.args = get_training_args()
        self.transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.PILToTensor(),
            transforms.Resize(
                (256,256),
                antialias=True,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ConvertImageDtype(torch.float32),

        ])

    def test_padim_mvtec_mobilenetv2(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = ["features.4", "features.7", "features.10"]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_resnet50(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.model_checkpoint_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_phinet(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'phinet_1.2_0.5_6_downsampling'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_micronet(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'micronet-m3'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_mcunet(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mcunet'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)



    def test_padim_realiad_mobilenetv2(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = ["features.4", "features.7", "features.10"]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path)

    def test_padim_realiad_resnet50(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path)

    def test_padim_realiad_phinet(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'phinet_1.2_0.5_6_downsampling'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path)

    def test_padim_realiad_micronet(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'micronet-m3'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path)

    def test_padim_realiad_mcunet(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'mcunet'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path)

    def test_padim_visa_mobilenetv2(self):
        pass

    def test_padim_visa_resnet50(self):
        pass

    def test_padim_visa_phinet(self):
        pass

    def test_padim_visa_micronet(self):
        pass

    def test_padim_visa_mcunet(self):
        pass

class PadimBenchmarkContaminated(unittest.TestCase):
    def setUp(self):
        self.seed = 3
        self.epoch = 10
        torch.manual_seed(self.seed)
        self.args = PadimArgs()
        self.args.contamination_ratio = 0.2
        self.args.batch_size = 2
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
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
        if self.args.contamination_ratio > 0:
            self.args.train_dataset.contaminate(self.args.test_dataset, self.args.contamination_ratio)
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
                "contamination_ratio": self.args.contamination_ratio
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
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.category = self.args.train_dataset.class_name
        if self.args.contamination_ratio > 0:
            self.args.train_dataset.contaminate(self.args.test_dataset, self.args.contamination_ratio)
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
                "contamination_ratio": self.args.contamination_ratio
            }, allow_val_change=True)
            self.logger.tags = ["padim", type(self.args.train_dataset).__name__, self.args.backbone]
            if self.args.contamination_ratio > 0:
                self.logger.tags += tuple(["contaminated"])
            self.logger.name = f"padim_{type(self.args.train_dataset).__name__}_{self.args.backbone}"
            train_padim(self.args, self.logger)
            self.logger.finish()

    def test_padim_mvtec_mobilenetv2_contaminated(self):
        self.logger.name = self._testMethodName

        self.logger.tags = ['padim', 'mvtec', 'mobilenetv2', 'contaminated']
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = ["features.4", "features.7", "features.10"]
        self.args.save_path = f"./"
        self.args.model_checkpoint_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        torch.cuda.empty_cache()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)

    def test_padim_mvtec_resnet50_contaminated(self):
        self.logger.name = self._testMethodName

        self.logger.tags = ['padim', 'mvtec', 'resnet50', 'contaminated']
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./"
        self.args.model_checkpoint_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
            gt_mask_size=IMAGE_SIZE
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        torch.cuda.empty_cache()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_phinet_contaminated(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'phinet_1.2_0.5_6_downsampling'
        self.args.ad_layers = [2, 6, 7]
        self.args.save_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)
        self.logger.tags = ['padim', 'mvtec', 'phinet', 'contaminated']
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)

    def test_padim_mvtec_micronet_contaminated(self):
        self.logger.tags = ['padim', 'mvtec', 'micronet', 'contaminated']
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'micronet-m1'
        self.args.ad_layers = [2, 4, 5]
        self.args.save_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)

    def test_padim_mvtec_mcunet_contaminated(self):
        self.logger.tags = ['padim', 'mvtec', 'micronet', 'contaminated']
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mcunet-in3'
        self.args.ad_layers = [3, 6, 9]
        self.args.save_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=IMAGE_SIZE,
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=IMAGE_SIZE,
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)



    def test_padim_realiad_mobilenetv2_contaminated(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = ["features.4", "features.7", "features.10"]
        self.args.save_path = f"./"
        self.args.model_checkpoint_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)
        self.logger.tags = ['padim', 'realiad', 'mobilenetv2', 'contaminated']
        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path)

    def test_padim_realiad_resnet50_contaminated(self):
        self.logger.name = self._testMethodName

        self.logger.tags = ['padim', 'realiad', 'resnet50', 'contaminated']
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path,
            logger=self.logger)

    def test_padim_realiad_phinet_contaminated(self):
        self.logger.name = self._testMethodName

        self.logger.tags = ['padim', 'realiad', 'phinet', 'contaminated']
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'phinet_1.2_0.5_6_downsampling'
        self.args.ad_layers = [2, 6, 7]
        self.args.save_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)
        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path,
            logger=self.logger)

    def test_padim_realiad_micronet_contaminated(self):
        self.logger.name = self._testMethodName

        self.logger.tags = ['padim', 'realiad', 'micronet', 'contaminated']
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'micronet-m1'
        self.args.ad_layers = [2, 4, 5]
        self.args.save_path = f"./pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)
        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path,
            logger=self.logger)

    def test_padim_realiad_mcunet_contaminated(self):
        self.logger.name = self._testMethodName

        self.logger.tags = ['padim', 'realiad', 'mcunet', 'contaminated']
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'mcunet-in3'
        self.args.ad_layers = [3, 6, 9]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
        }, allow_val_change=True)
        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=IMAGE_SIZE,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path,
            logger=self.logger)

    def test_padim_visa_mobilenetv2_contaminated(self):
        self.logger.tags = ['padim', 'visa', 'mobilenetv2', 'contaminated']
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = ["features.4", "features.7", "features.10"]
        self.args.save_path = f"./"
        self.args.model_checkpoint_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        }, allow_val_change=True)

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=IMAGE_SIZE,
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)

    def test_padim_visa_resnet50_contaminated(self):
        self.logger.tags = ['padim', 'visa', 'resnet50', 'contaminated']
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./"
        self.args.model_checkpoint_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        }, allow_val_change=True)

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=IMAGE_SIZE,
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)

    def test_padim_visa_phinet_contaminated(self):
        self.logger.tags = ['padim', 'visa', 'phinet', 'contaminated']
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'phinet_1.2_0.5_6_downsampling'
        self.args.ad_layers = [2, 6, 7]
        self.args.save_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        }, allow_val_change=True)

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TEST, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=IMAGE_SIZE,
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)

        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)

    def test_padim_visa_micronet_contaminated(self):
        self.logger.tags = ['padim', 'visa', 'micronet', 'contaminated']
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'micronet-m1'
        self.args.ad_layers = [2, 4, 5]
        self.args.save_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        }, allow_val_change=True)

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=IMAGE_SIZE,
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)

        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)

    def test_padim_visa_mcunet_contaminated(self):
        self.logger.tags = ['padim', 'visa', 'mcunet', 'contaminated']
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mcunet-in3'
        self.args.ad_layers = [3, 6, 9]
        self.args.save_path = f"./"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        }, allow_val_change=True)

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=IMAGE_SIZE,
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)

        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path,
                    logger=self.logger)


if __name__ == '__main__':
    unittest.main()
