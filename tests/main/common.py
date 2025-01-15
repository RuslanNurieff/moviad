import unittest
from dataclasses import dataclass
from typing import List

import torch
from sympy import false

from main_scripts.main_patchcore import train_patchcore, test_patchcore, train_patchcore, test_patchcore
MODE = "train"
CATEGORY = "pill"
BACKBONE = "mobilenet_v2"
AD_LAYERS = ["features.4", "features.7", "features.10"]
SAVE_PATH = "./patch.pt"
VISUAL_TEST_PATH = "./visual_test"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
MODEL_CHECKPOINT_PATH = "./"
MAX_DATASET_SIZE = 500
EPOCHS = 1
INPUT_SIZES = {
        "mcunet-in3": (176, 176),
        "mobilenet_v2": (224, 224),
        "phinet_1.2_0.5_6_downsampling": (224, 224),
        "wide_resnet50_2": (224, 224),
        "micronet-m1": (224, 224),
    }
