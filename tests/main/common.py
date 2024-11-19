import unittest
from dataclasses import dataclass
from typing import List

import torch
from sympy import false

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
INPUT_SIZES = {
        "mcunet-in3": (176, 176),
        "mobilenet_v2": (224, 224),
        "phinet_1.2_0.5_6_downsampling": (224, 224),
        "wide_resnet50_2": (224, 224),
        "micronet-m1": (224, 224),
    }

class StfpmTestParams:
    feature_maps_dir = None
    boot_layer = None
    categories: list[str] = None
    model_name: str = None

    input_size: int = None
    ad_model: str = None
    output_size: tuple = None
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size: int = 32
    test_dataset: torch.utils.data.Dataset = None
    input_sizes: dict = INPUT_SIZES
    results_dirpath: str = './results'
    checkpoint_dir: str = './checkpoints'
    trained_models_filepaths: list = None
    class_name: str = None


class StfpmTrainingParams:
    def __init__(self, train_dataset, categories, ad_layers, epochs: int, seeds: list[int], batch_size: int, model_name: str, results_dirpath, device,
                 input_sizes, output_size, checkpoint_dir, dataset_path, disable_dataset_norm, log_dirpath,
                 boot_layer=None):
        self.train_dataset = train_dataset
        self.categories = categories
        self.ad_layers = ad_layers
        self.results_dirpath = results_dirpath
        self.epochs = epochs
        self.seeds = seeds
        self.batch_size = batch_size
        self.backbone_model_name = model_name
        self.device = device
        self.img_input_size = input_sizes[model_name]
        self.img_output_size = output_size
        self.early_stopping = None  # 0.01 | None | 0.002
        self.student_bootstrap_layer = [boot_layer] if boot_layer is not None else [False]
        self.checkpoint_dir = checkpoint_dir
        self.normalize_dataset = bool(not disable_dataset_norm)
        self.dataset_path = dataset_path
        self.log_dirpath = log_dirpath


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

def get_stfpm_test_args():
    return StfpmTestParams()

def get_stfpm_train_args():
    return StfpmTrainingParams(
        train_dataset=None,
        categories=['pill'],
        ad_layers=[[8, 9], [10, 11, 12]],
        results_dirpath='./results',
        epochs=10,
        seeds=[42],
        batch_size=32,
        model_name='mobilenet_v2',
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        input_sizes=INPUT_SIZES,
        output_size=(224, 224),
        checkpoint_dir='./checkpoints',
        log_dirpath='./logs',
        dataset_path=REALIAD_DATASET_PATH,
        disable_dataset_norm=false,
        boot_layer=0
    )


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
