import unittest
from dataclasses import dataclass
from typing import List

import torch

from main_scripts.main_patchcore import train_patchcore, test_patchcore, train_patchcore_2, test_patchcore_2

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
EPOCHS = 1

@dataclass
class TrainingArguments:
    """
    Arguments for training and testing.

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
    epochs: int
    visual_test_path: str
    device: torch.device
    seed: int
    model_checkpoint_path: str = None


def get_training_args():
    return TrainingArguments(
        mode=MODE,
        dataset_path='',
        category=CATEGORY,
        backbone=BACKBONE,
        ad_layers=AD_LAYERS,
        save_path=SAVE_PATH,
        epochs=EPOCHS,
        visual_test_path=VISUAL_TEST_PATH,
        device=DEVICE,
        seed=SEED,
        model_checkpoint_path=MODEL_CHECKPOINT_PATH
    )
