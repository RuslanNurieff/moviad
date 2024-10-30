import unittest
from dataclasses import dataclass
from typing import List

import torch

from main_scripts.main_patchcore import train_patchcore, test_patchcore

MVTECH_DATASET_PATH = 'E:\\VisualAnomalyDetection\\datasets\\mvtec'
REALIAD_DATASET_PATH = 'E:\\VisualAnomalyDetection\\datasets\\Real-IAD\\realiad_256'
# Global variables for arguments
MODE = "train"
CATEGORY = "pill"
BACKBONE = "mobilenet_v2"
AD_LAYERS = ["features.4", "features.7", "features.10"]
SAVE_PATH = "./patch.pt"
VISUAL_TEST_PATH = "./visual_test"
DEVICE = torch.device("cuda:0")
SEED = 1
MODEL_CHECKPOINT_PATH = "./patch.pt"




@dataclass
class PatchCoreTrainingArguments:
    """
    Arguments for training PatchCore

    Example: python main_scripts/main_patchcore.py --mode train
                        --dataset_path /home/datasets/mvtec
                        --category pill --backbone mobilenet_v2
                        --ad_layers features.4 features.7 features.10
                        --device cuda:0
                        --save_path ./patch.pt
    """
    mode: str
    dataset_path: str
    category: str
    backbone: str
    ad_layers: List[str]
    save_path: str
    visual_test_path: str
    device: torch.device
    seed: int
    model_checkpoint_path: str = None


class PatchCoreTrainTests(unittest.TestCase):

    def setUp(self):
        self.args = PatchCoreTrainingArguments(
            mode=MODE,
            dataset_path='',
            category=CATEGORY,
            backbone=BACKBONE,
            ad_layers=AD_LAYERS,
            save_path=SAVE_PATH,
            visual_test_path=VISUAL_TEST_PATH,
            device=DEVICE,
            seed=SEED
        )

    def test_patchcore_train_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        train_patchcore(self.args.dataset_path, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device)

    def test_patchcore_train_with_realiad_dataset(self):
        self.args.dataset_path = REALIAD_DATASET_PATH
        train_patchcore(self.args.dataset_path, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.save_path, self.args.device)


class PatchCoreInferenceTests(unittest.TestCase):
    def setUp(self):
        self.args = PatchCoreTrainingArguments(
            mode=MODE,
            dataset_path='',
            category=CATEGORY,
            backbone=BACKBONE,
            ad_layers=AD_LAYERS,
            save_path=SAVE_PATH,
            visual_test_path=VISUAL_TEST_PATH,
            device=DEVICE,
            seed=SEED,
            model_checkpoint_path=MODEL_CHECKPOINT_PATH
        )

    def test_patchcore_inference_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        test_patchcore(self.args.dataset_path, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.model_checkpoint_path, self.args.device)

    def test_patchcore_inference_with_realiad_dataset(self):
        self.args.dataset_path = REALIAD_DATASET_PATH
        test_patchcore(self.args.dataset_path, self.args.category, self.args.backbone, self.args.ad_layers,
                        self.args.model_checkpoint_path, self.args.device)


if __name__ == '__main__':
    unittest.main()
