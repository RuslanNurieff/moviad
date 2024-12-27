import unittest
from dataclasses import dataclass
from tkinter import Image
from tkinter.tix import IMAGE
from typing import List

import torch
from torchvision.transforms import transforms

from main_scripts.main_patchcore import train_patchcore, test_patchcore, train_patchcore, test_patchcore, \
    train_patchcore, IMAGE_SIZE, REAL_IAD_DATASET_PATH, AUDIO_JACK_DATASET_JSON, test_patchcore
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum
from moviad.utilities.configurations import TaskType, Split
from tests.main.common import TrainingArguments

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

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float32),
])


class PatchCoreTrainTests(unittest.TestCase):

    def setUp(self):
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

    def test_patchcore_train_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        train_dataset = MVTecDataset(TaskType.SEGMENTATION, self.args.dataset_path, self.args.category, "train")
        test_dataset = MVTecDataset(TaskType.SEGMENTATION, self.args.dataset_path, self.args.category, "test")
        train_patchcore(train_dataset, test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device)

    def test_patchcore_train_with_realiad_dataset(self):
        self.args.dataset_path = REALIAD_DATASET_PATH

        train_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK,
                                       self.args.dataset_path,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=transform)

        test_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK,
                                      self.args.dataset_path,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=IMAGE_SIZE,
                                      gt_mask_size=IMAGE_SIZE,
                                      transform=transform)

        train_patchcore(train_dataset, test_dataset, 'audiojack', self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device)



class PatchCoreInferenceTests(unittest.TestCase):
    def setUp(self):
        self.args = TrainingArguments(
            mode=MODE,
            dataset_path='',
            category=CATEGORY,
            backbone=BACKBONE,
            ad_layers=AD_LAYERS,
            epochs=100,
            save_path=SAVE_PATH,
            visual_test_path=VISUAL_TEST_PATH,
            device=DEVICE,
            seed=SEED,
            model_checkpoint_path=MODEL_CHECKPOINT_PATH
        )

    def test_patchcore_inference_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        test_dataset = MVTecDataset(TaskType.SEGMENTATION, self.args.dataset_path, self.args.category, "test")
        test_patchcore(test_dataset, self.args.category, self.args.backbone, self.args.ad_layers,
                       self.args.model_checkpoint_path, self.args.device)

    def test_patchcore_inference_with_realiad_dataset(self):
        self.args.dataset_path = REALIAD_DATASET_PATH
        test_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK,
                                      self.args.dataset_path,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=IMAGE_SIZE,
                                      gt_mask_size=IMAGE_SIZE,
                                      transform=transform)
        test_patchcore(test_dataset, 'audiojack', self.args.backbone, self.args.ad_layers,
                       self.args.model_checkpoint_path, self.args.device)


if __name__ == '__main__':
    unittest.main()
