import unittest
import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode

from main_scripts.main_padim import main_train_padim
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass
from moviad.entrypoints.padim import train_padim
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_tests import REAL_IAD_DATASET_PATH, AUDIO_JACK_DATASET_JSON
from tests.main.common import TrainingArguments, get_training_args, MVTECH_DATASET_PATH


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
            img_size=(256, 256),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
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
            img_size=(256, 256),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
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
            img_size=(256, 256),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
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
            img_size=(256, 256),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_mcunet(self):
        pass

    def test_padim_mvtec_resnet18(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'resnet18'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=(256, 256),
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
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
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.model_checkpoint_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=(256, 256),
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=(256, 256),
                                        gt_mask_size=(256, 256),
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path)

    def test_padim_realiad_resnet50(self):
        pass

    def test_padim_realiad_phinet(self):
        pass

    def test_padim_realiad_micronet(self):
        pass

    def test_padim_realiad_mcunet(self):
        pass

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
        self.args = get_training_args()
        self.seed = 3
        torch.manual_seed(self.seed)
        self.contamination_ratio = 0.1
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
        wandb.setup(wandb.Settings(program=__name__, program_relpath=__name__))
        self.logger = wandb.init(project="moviad_benchmark")

    def test_padim_mvtec_mobilenetv2_contaminated(self):
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
            img_size=(256, 256),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_dataset.contaminate(self.contamination_ratio)
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_resnet50_contaminated(self):
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
            img_size=(256, 256),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_phinet_contaminated(self):
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
            img_size=(256, 256),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_micronet_contaminated(self):
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
            img_size=(256, 256),
        )
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_mvtec_mcunet_contaminated(self):
        pass

    def test_padim_mvtec_resnet18_contaminated(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.category = 'pill'
        self.args.backbone = 'resnet18'
        self.args.ad_layers = ["layer1", "layer2", "layer3"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TRAIN,
            img_size=(256, 256),
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
        )
        train_dataset.load_dataset()
        test_dataset.load_dataset()
        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,
                    self.args.model_checkpoint_path)

    def test_padim_realiad_mobilenetv2_contaminated(self):
        self.args.dataset_path = REAL_IAD_DATASET_PATH
        self.args.category = RealIadClass.AUDIOJACK
        self.args.backbone = 'mobilenet_v2'
        self.args.ad_layers = ["features.4", "features.7", "features.10"]
        self.args.save_path = f"./{self._testMethodName}/patch.pt"
        self.args.model_checkpoint_path = f"./{self._testMethodName}/patch.pt"
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        train_dataset = RealIadDataset(self.args.category,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=(256, 256),
                                       transform=self.transform)

        test_dataset = RealIadDataset(self.args.category,
                                        REAL_IAD_DATASET_PATH,
                                        AUDIO_JACK_DATASET_JSON,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TEST,
                                        image_size=(256, 256),
                                        gt_mask_size=(256, 256),
                                        transform=self.transform)
        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_padim(train_dataset, test_dataset, self.args.category, self.args.backbone,
            self.args.ad_layers,
            self.args.device,
            self.args.model_checkpoint_path)

    def test_padim_realiad_resnet50_contaminated(self):
        pass

    def test_padim_realiad_phinet_contaminated(self):
        pass

    def test_padim_realiad_micronet_contaminated(self):
        pass

    def test_padim_realiad_mcunet_contaminated(self):
        pass

    def test_padim_visa_mobilenetv2_contaminated(self):
        pass

    def test_padim_visa_resnet50_contaminated(self):
        pass

    def test_padim_visa_phinet_contaminated(self):
        pass

    def test_padim_visa_micronet_contaminated(self):
        pass

    def test_padim_visa_mcunet_contaminated(self):
        pass


if __name__ == '__main__':
    unittest.main()
