import unittest
import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode

from datasets.visadataset_tests import VISA_DATASET_PATH, VISA_DATASET_CSV_PATH
from main.common import TrainingArguments
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass
from moviad.entrypoints.patchcore import train_patchcore
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_tests import REAL_IAD_DATASET_PATH, AUDIO_JACK_DATASET_JSON

MVTECH_DATASET_PATH = 'E:\\VisualAnomalyDetection\\datasets\\mvtec'
REALIAD_DATASET_PATH = 'E:\\VisualAnomalyDetection\\datasets\\Real-IAD\\realiad_256'
MODE = "train"
CATEGORY = "pill"
BACKBONE = "mobilenet_v2"
AD_LAYERS = ["features.4", "features.7", "features.10"]
SAVE_PATH = "./patch.pt"
VISUAL_TEST_PATH = "./visual_test"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
MODEL_CHECKPOINT_PATH = "./patch.pt"
MAX_DATASET_SIZE = 500
IMAGE_SIZE = (224, 224)

class PatchCoreBenchmarkContaminated(unittest.TestCase):
    def setUp(self):
        self.seed = 42
        self.contamination_ratio = 0.1
        self.args = TrainingArguments(
            mode=MODE,
            dataset_path='',
            category=CATEGORY,
            backbone=BACKBONE,
            epochs=100,
            ad_layers=AD_LAYERS,
            save_path=SAVE_PATH,
            visual_test_path=VISUAL_TEST_PATH,
            device=DEVICE,
            seed=SEED
        )
        torch.manual_seed(self.seed)
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
        self.logger = wandb.init(project="moviad_benchmark", group="patchcore_contaminated")

    def test_patchcore_mvtec_mobilenetv2_contaminated(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = ['features.4', 'features.7', 'features.10']
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

        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset

        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)

    def test_patchcore_mvtec_resnet50_contaminated(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'wide_resnet50_2'
        self.args.ad_layers = [[1, 2, 3]]
        self.args.save_path = f"./patch.pt"
        self.args.model_checkpoint_path = f"./patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.img_input_size = IMAGE_SIZE
        self.args.img_output_size = IMAGE_SIZE
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)

    def test_patchcore_mvtec_phinet_contamianted(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'phinet_1.2_0.5_6_downsampling'
        self.args.ad_layers = [2, 6, 7] # [4, 5, 6] , [5, 6, 7] (PaSTE), [6, 7, 8]
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)

    def test_patchcore_mvtec_micronet_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)

    def test_patchcore_mvtec_mcunet_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_mvtec_resnet18_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_realiad_mobilenetv2_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_realiad_resnet50_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_realiad_phinet_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_realiad_micronet_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_realiad_mcunet_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)

    def test_patchcore_visa_mobilenetv2_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_visa_resnet50_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_visa_phinet_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_visa_micronet_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)


    def test_patchcore_visa_mcunet_contaminated(self):
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
        self.args.train_dataset = train_dataset
        self.args.test_dataset = test_dataset
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device, self.logger)

if __name__ == '__main__':
    unittest.main()
