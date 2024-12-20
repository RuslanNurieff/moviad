import unittest
import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode
import tempfile

from datasets.visadataset_tests import VISA_DATASET_PATH, VISA_DATASET_CSV_PATH
from main_scripts.main_padim import main_train_padim
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.entrypoints.stfpm import train_stfpm, test_stfpm
from moviad.entrypoints.padim import train_padim
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_tests import REAL_IAD_DATASET_PATH, AUDIO_JACK_DATASET_JSON
from tests.main.common import TrainingArguments, get_training_args, MVTECH_DATASET_PATH, get_stfpm_train_args
import shutil


class StfpmBenchmark(unittest.TestCase):
    def setUp(self):
        self.seed = 3
        torch.manual_seed(self.seed)
        self.args = get_training_args()
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
        self.logger = wandb.init(project="moviad_benchmark")


def test_Stfpm_mvtec_mobilenetv2(self):
    self.logger.tags = ["stfpm", "mvtec", "mobilenet_v2"]
    self.args.dataset_path = MVTECH_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'mobilenet_v2'
    self.args.ad_layers = ["features.4", "features.7", "features.10"]
    self.args.save_path = f"./patch.pt"
    self.args.model_checkpoint_path = f"./patch.pt"

    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "mvtec",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
    })

    train_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TRAIN,
        img_size=(224, 224),
    )
    test_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TEST,
        img_size=(224, 224),
    )
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset, 32, self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_mvtec_resnet50(self):
    self.logger.tags = ["stfpm", "mvtec", "wide_resnet50_2"]
    self.args.dataset_path = MVTECH_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'wide_resnet50_2'
    self.args.ad_layers = ["layer1", "layer2", "layer3"]
    self.args.save_path = f"./{self._testMethodName}/patch.pt"
    self.args.model_checkpoint_path = f"./{self._testMethodName}/patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.config.update({
        "dataset": "mvtec",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
    })
    self.logger.name = self._testMethodName
    train_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TRAIN,
        img_size=(224, 224),
    )
    test_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TEST,
        img_size=(224, 224),
    )
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_mvtec_phinet(self):
    self.logger.tags = ["stfpm", "mvtec", "phinet_1.2_0.5_6_downsampling"]
    self.args.dataset_path = MVTECH_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'phinet_1.2_0.5_6_downsampling'
    self.args.ad_layers = [2, 6, 7]
    self.args.save_path = f"./patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "mvtec",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TRAIN,
        img_size=(224, 224),
    )

    test_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TEST,
        img_size=(224, 224),
    )

    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                4,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_mvtec_micronet(self):
    self.logger.tags = ["stfpm", "mvtec", "micronet-m1"]
    self.args.dataset_path = MVTECH_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'micronet-m1'
    self.args.ad_layers = [1, 2, 3]
    self.args.save_path = f"./patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "mvtec",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TRAIN,
        img_size=(224, 224),
    )
    test_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TEST,
        img_size=(224, 224),
    )
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_mvtec_mcunet(self):
    self.logger.tags = ["stfpm", "mvtec", "mcunet-in3"]
    self.args.dataset_path = MVTECH_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'mcunet-in3'
    self.args.ad_layers = ["layer1", "layer2", "layer3"]
    self.args.save_path = f"./patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "mvtec",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TRAIN,
        img_size=(224, 224),
    )

    test_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TEST,
        img_size=(224, 224),
    )

    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_mvtec_resnet18(self):
    self.logger.tags = ["stfpm", "mvtec", "resnet18"]
    self.args.dataset_path = MVTECH_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'resnet18'
    self.args.ad_layers = ["layer1", "layer2", "layer3"]
    self.args.save_path = f"./patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "mvtec",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TRAIN,
        img_size=(224, 224),
    )

    test_dataset = MVTecDataset(
        TaskType.SEGMENTATION,
        self.args.dataset_path,
        self.args.category,
        Split.TEST,
        img_size=(224, 224),
    )
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_realiad_mobilenetv2(self):
    self.logger.tags = ["stfpm", "realiad", "mobilenet_v2"]
    self.args.dataset_path = REAL_IAD_DATASET_PATH
    self.args.category = RealIadClass.AUDIOJACK
    self.args.backbone = 'mobilenet_v2'
    self.args.ad_layers = ["features.4", "features.7", "features.10"]
    self.args.save_path = f"./{self._testMethodName}/patch.pt"
    self.args.model_checkpoint_path = f"./{self._testMethodName}/patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "realiad",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = RealIadDataset(self.args.category,
                                   REAL_IAD_DATASET_PATH,
                                   AUDIO_JACK_DATASET_JSON,
                                   task=TaskType.SEGMENTATION,
                                   split=Split.TRAIN,
                                   image_size=(224, 224),
                                   transform=self.transform)

    test_dataset = RealIadDataset(self.args.category,
                                  REAL_IAD_DATASET_PATH,
                                  AUDIO_JACK_DATASET_JSON,
                                  task=TaskType.SEGMENTATION,
                                  split=Split.TEST,
                                  image_size=(224, 224),
                                  gt_mask_size=(224, 224),
                                  transform=self.transform)
    train_dataset.load_dataset()
    test_dataset.load_dataset()

    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_realiad_resnet50(self):
    self.logger.tags = ["stfpm", "realiad", "wide_resnet50_2"]
    self.args.dataset_path = REAL_IAD_DATASET_PATH
    self.args.category = RealIadClass.AUDIOJACK
    self.args.backbone = 'wide_resnet50_2'
    self.args.ad_layers = ["layer1", "layer2", "layer3"]
    self.args.save_path = f"./{self._testMethodName}/patch.pt"
    self.args.model_checkpoint_path = f"./{self._testMethodName}/patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "realiad",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = RealIadDataset(self.args.category,
                                   REAL_IAD_DATASET_PATH,
                                   AUDIO_JACK_DATASET_JSON,
                                   task=TaskType.SEGMENTATION,
                                   split=Split.TRAIN,
                                   image_size=(224, 224),
                                   transform=self.transform)

    test_dataset = RealIadDataset(self.args.category,
                                  REAL_IAD_DATASET_PATH,
                                  AUDIO_JACK_DATASET_JSON,
                                  task=TaskType.SEGMENTATION,
                                  split=Split.TEST,
                                  image_size=(224, 224),
                                  gt_mask_size=(224, 224),
                                  transform=self.transform)
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def Stfpm_realiad_phinet(self):
    self.logger.tags = ["stfpm", "realiad", "phinet_2.3_0.75_5"]
    self.args.dataset_path = REAL_IAD_DATASET_PATH
    self.args.category = RealIadClass.AUDIOJACK
    self.args.backbone = 'phinet_2.3_0.75_5'
    self.args.ad_layers = ["1", "2", "3"]
    self.args.save_path = f"./{self._testMethodName}/patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "realiad",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = RealIadDataset(self.args.category,
                                   REAL_IAD_DATASET_PATH,
                                   AUDIO_JACK_DATASET_JSON,
                                   task=TaskType.SEGMENTATION,
                                   split=Split.TRAIN,
                                   image_size=(224, 224),
                                   transform=self.transform)

    test_dataset = RealIadDataset(self.args.category,
                                  REAL_IAD_DATASET_PATH,
                                  AUDIO_JACK_DATASET_JSON,
                                  task=TaskType.SEGMENTATION,
                                  split=Split.TEST,
                                  image_size=(224, 224),
                                  gt_mask_size=(224, 224),
                                  transform=self.transform)
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_realiad_micronet(self):
    self.logger.tags = ["stfpm", "realiad", "micronet-m1"]
    self.args.dataset_path = REAL_IAD_DATASET_PATH
    self.args.category = RealIadClass.AUDIOJACK
    self.args.backbone = 'micronet-m1'
    self.args.ad_layers = [1, 2, 3]
    self.args.save_path = f"./{self._testMethodName}/patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "realiad",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = RealIadDataset(self.args.category,
                                   REAL_IAD_DATASET_PATH,
                                   AUDIO_JACK_DATASET_JSON,
                                   task=TaskType.SEGMENTATION,
                                   split=Split.TRAIN,
                                   image_size=(224, 224),
                                   transform=self.transform)

    test_dataset = RealIadDataset(self.args.category,
                                  REAL_IAD_DATASET_PATH,
                                  AUDIO_JACK_DATASET_JSON,
                                  task=TaskType.SEGMENTATION,
                                  split=Split.TEST,
                                  image_size=(224, 224),
                                  gt_mask_size=(224, 224),
                                  transform=self.transform)
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_realiad_mcunet(self):
    self.logger.tags = ["stfpm", "realiad", "mcunet-in3"]
    self.args.dataset_path = REAL_IAD_DATASET_PATH
    self.args.category = RealIadClass.AUDIOJACK
    self.args.backbone = 'mcunet-in3'
    self.args.ad_layers = ["layer1", "layer2", "layer3"]
    self.args.save_path = f"./{self._testMethodName}/patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "realiad",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })


def test_Stfpm_visa_mobilenetv2(self):
    self.logger.tags = ["stfpm", "visa", "mobilenet_v2"]
    self.args.dataset_path = VISA_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'mobilenet_v2'
    self.args.ad_layers = ["features.4", "features.7", "features.10"]
    self.args.save_path = f"./patch.pt"
    self.args.model_checkpoint_path = f"./patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "visa",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = VisaDataset(VISA_DATASET_PATH,
                                VISA_DATASET_CSV_PATH,
                                Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                gt_mask_size=(224, 224),
                                transform=self.transform)

    test_dataset = VisaDataset(VISA_DATASET_PATH,
                               VISA_DATASET_CSV_PATH,
                               Split.TEST, VisaDatasetCategory.pipe_fryum,
                               gt_mask_size=(224, 224),
                               transform=self.transform)
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_visa_resnet50(self):
    self.logger.tags = ["stfpm", "visa", "wide_resnet50_2"]
    self.args.dataset_path = VISA_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'wide_resnet50_2'
    self.args.ad_layers = ["layer1", "layer2", "layer3"]
    self.args.save_path = f"./{self._testMethodName}/patch.pt"
    self.args.model_checkpoint_path = f"./{self._testMethodName}/patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "visa",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = VisaDataset(VISA_DATASET_PATH,
                                VISA_DATASET_CSV_PATH,
                                Split.TEST, VisaDatasetCategory.pipe_fryum,
                                gt_mask_size=(224, 224),
                                transform=self.transform)

    test_dataset = VisaDataset(VISA_DATASET_PATH,
                               VISA_DATASET_CSV_PATH,
                               Split.TEST, VisaDatasetCategory.pipe_fryum,
                               gt_mask_size=(224, 224),
                               transform=self.transform)
    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def Stfpm_visa_phinet(self):
    self.logger.tags = ["stfpm", "visa", "phinet_2.3_0.75_5"]
    self.args.dataset_path = VISA_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'phinet_2.3_0.75_5'
    self.args.ad_layers = ["1", "2", "3"]
    self.args.save_path = f"./patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "visa",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = VisaDataset(VISA_DATASET_PATH,
                                VISA_DATASET_CSV_PATH,
                                Split.TEST, VisaDatasetCategory.pipe_fryum,
                                gt_mask_size=(224, 224),
                                transform=self.transform)

    test_dataset = VisaDataset(VISA_DATASET_PATH,
                               VISA_DATASET_CSV_PATH,
                               Split.TEST, VisaDatasetCategory.pipe_fryum,
                               gt_mask_size=(224, 224),
                               transform=self.transform)

    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_visa_micronet(self):
    self.logger.tags = ["stfpm", "visa", "micronet-m1"]
    self.args.dataset_path = VISA_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'micronet-m1'
    self.args.ad_layers = [1, 2, 3]
    self.args.save_path = f"./patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "visa",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = VisaDataset(VISA_DATASET_PATH,
                                VISA_DATASET_CSV_PATH,
                                Split.TEST, VisaDatasetCategory.pipe_fryum,
                                gt_mask_size=(224, 224),
                                transform=self.transform)

    test_dataset = VisaDataset(VISA_DATASET_PATH,
                               VISA_DATASET_CSV_PATH,
                               Split.TEST, VisaDatasetCategory.pipe_fryum,
                               gt_mask_size=(224, 224),
                               transform=self.transform)

    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


def test_Stfpm_visa_mcunet(self):
    self.logger.tags = ["stfpm", "visa", "mcunet-in3"]
    self.args.dataset_path = VISA_DATASET_PATH
    self.args.category = 'pill'
    self.args.backbone = 'mcunet-in3'
    self.args.ad_layers = ["layer1", "layer2", "layer3"]
    self.args.save_path = f"./patch.pt"
    self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.logger.name = self._testMethodName
    self.logger.config.update({
        "dataset": "visa",
        "category": self.args.category,
        "backbone": self.args.backbone,
        "ad_layers": self.args.ad_layers,
        "seed": self.seed
    })

    train_dataset = VisaDataset(VISA_DATASET_PATH,
                                VISA_DATASET_CSV_PATH,
                                Split.TEST, VisaDatasetCategory.pipe_fryum,
                                gt_mask_size=(224, 224),
                                transform=self.transform)

    test_dataset = VisaDataset(VISA_DATASET_PATH,
                               VISA_DATASET_CSV_PATH,
                               Split.TEST, VisaDatasetCategory.pipe_fryum,
                               gt_mask_size=(224, 224),
                               transform=self.transform)

    train_dataset.load_dataset()
    test_dataset.load_dataset()
    train_stfpm(train_dataset, test_dataset,
                32,
                self.args.category,
                self.args.backbone,
                self.args.ad_layers,
                30,
                self.args.save_path,
                self.args.device, self.logger)


class StfpmBenchmarkContaminated(unittest.TestCase):
    def setUp(self):
        self.seed = 3
        self.epoch = 10
        torch.manual_seed(self.seed)
        self.contamination_ratio = 0.1
        self.args = get_stfpm_train_args()
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
        self.logger = wandb.init(project="moviad_benchmark", group="stfpm_contaminated")

    def test_Stfpm_mvtec_mobilenetv2_contaminated(self):
        self.logger.tags = ["stfpm", "mvtec", "mobilenet_v2", "contaminated"]
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = [[4, 7, 10]]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "mvtec",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "contamination_ratio": self.contamination_ratio,
        })

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=(224, 224),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(224, 224),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)

        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset

        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_mvtec_resnet50_contaminated(self):
        self.logger.tags = ["stfpm", "mvtec", "mobilenet_v2", "contaminated"]
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.img_input_size = (224, 224)
        self.args.img_output_size = (224, 224)
        self.logger.config.update({
            "dataset": "mvtec",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "contamination_ratio": self.contamination_ratio,
        })
        self.logger.name = self._testMethodName
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=(224, 224),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(224, 224),
            gt_mask_size=(224, 224)
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_mvtec_phinet_contamianted(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'phinet_1.2_0.5_6_downsampling'
        self.args.ad_layers = [2, 6, 7]  # [4, 5, 6] , [5, 6, 7] (PaSTE), [6, 7, 8]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "mvtec",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio,
        })

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=(224, 224),
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(224, 224),
        )

        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_mvtec_micronet_contaminated(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'micronet-m1'
        self.args.ad_layers = [1, 2, 3]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "mvtec",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio,
        })

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=(224, 224),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(224, 224),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_mvtec_mcunet_contaminated(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mcunet-in3'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "mvtec",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio,
        })

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=(224, 224),
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(224, 224),
        )

        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_mvtec_resnet18_contaminated(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'resnet18'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "mvtec",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=(224, 224),
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(224, 224),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_realiad_mobilenetv2_contaminated(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = [[4, 7, 10]]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=(224, 224),
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=(224, 224),
                                      gt_mask_size=(224, 224),
                                      transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_realiad_resnet50_contaminated(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=(224, 224),
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=(224, 224),
                                      gt_mask_size=(224, 224),
                                      transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)

    def test_Stfpm_realiad_phinet_contaminated(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'phinet_2.3_0.75_5'
        self.args.ad_layers = [2, 6, 7]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=(224, 224),
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=(224, 224),
                                      gt_mask_size=(224, 224),
                                      transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)

    def test_Stfpm_realiad_micronet_contaminated(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'micronet-m1'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=(224, 224),
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=(224, 224),
                                      gt_mask_size=(224, 224),
                                      transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_realiad_mcunet_contaminated(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'mcunet-in3'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "realiad",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=(224, 224),
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=(224, 224),
                                      gt_mask_size=(224, 224),
                                      transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_visa_mobilenetv2_contaminated(self):
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = [[4, 7, 10]]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=(224, 224),
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=(224, 224),
                                   transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_visa_resnet50_contaminated(self):
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio,
        }, allow_val_change=True)

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=(224, 224),
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=(224, 224),
                                   transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_visa_phinet_contaminated(self):
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'phinet_2.3_0.75_5'
        self.args.ad_layers = [[2, 6, 7]]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TEST, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=(224, 224),
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=(224, 224),
                                   transform=self.transform)

        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_visa_micronet_contaminated(self):
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'micronet-m1'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=(224, 224),
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=(224, 224),
                                   transform=self.transform)

        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)

    def test_Stfpm_visa_mcunet_contaminated(self):
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mcunet-in3'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.name = self._testMethodName
        self.logger.config.update({
            "dataset": "visa",
            "category": self.args.category,
            "backbone": self.args.backbone,
            "ad_layers": self.args.ad_layers,
            "seed": self.seed,
            "contamination_ratio": self.contamination_ratio
        })

        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                    VISA_DATASET_CSV_PATH,
                                    Split.TRAIN, VisaDatasetCategory.pipe_fryum,
                                    gt_mask_size=(224, 224),
                                    transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.pipe_fryum,
                                   gt_mask_size=(224, 224),
                                   transform=self.transform)

        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(test_dataset, self.contamination_ratio)
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_stfpm(self.args, self.logger)
        test_stfpm(self.args, self.logger)


if __name__ == '__main__':
    unittest.main()
