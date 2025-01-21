from benchmark_config import DatasetConfig
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.utilities.configurations import TaskType, Split
import unittest
from tokenize import group

import torch
import wandb
from torchvision.transforms import transforms, InterpolationMode
import tempfile
from main_scripts.main_padim import main_train_padim
from main_scripts.main_patchcore import IMAGE_SIZE
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum, REAL_IAD_CATEGORIES_JSONS
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.entrypoints.cfa import train_cfa
from moviad.entrypoints.padim import train_padim
from moviad.utilities.configurations import TaskType, Split
import shutil

config = DatasetConfig('./config.yaml')

REAL_IAD_JSON_ROOT_PATH = config.realiad_json_root_path

backbones = {
    "mobilenet_v2": ["features.4", "features.7", "features.10"],
    "wide_resnet50_2": ["layer1", "layer2", "layer3"],
    "phinet_1.2_0.5_6_downsampling": [2, 6, 7],
    "micronet-m1": [2, 4, 5],
    "mcunet-in3": [3, 6, 9],
    "resnet18": ["layer1", "layer2", "layer3"]
}

IMAGE_SIZE = (224, 224)

mvtec_train_dataset = MVTecDataset(
    TaskType.SEGMENTATION,
    config.mvtec_root_path,
    'pill',
    Split.TRAIN,
    img_size=IMAGE_SIZE,
)

mvtec_test_dataset = MVTecDataset(
    TaskType.SEGMENTATION,
    config.mvtec_root_path,
    'pill',
    Split.TEST,
    img_size=IMAGE_SIZE,
    gt_mask_size=IMAGE_SIZE,
)

real_iad_train_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK.value,
                                        config.realiad_root_path,
                                        config.realiad_json_root_path,
                                        task=TaskType.SEGMENTATION,
                                        split=Split.TRAIN,
                                        image_size=IMAGE_SIZE,
                                        transform=transforms.Compose([
                                            transforms.Resize(IMAGE_SIZE),
                                            transforms.PILToTensor(),
                                            transforms.Resize(
                                                IMAGE_SIZE,
                                                antialias=True,
                                                interpolation=InterpolationMode.NEAREST,
                                            ),
                                            transforms.ConvertImageDtype(torch.float32)
                                        ]))

real_iad_test_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK.value,
                                       config.realiad_root_path,
                                       config.realiad_json_root_path,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TEST,
                                       image_size=IMAGE_SIZE,
                                       gt_mask_size=IMAGE_SIZE,
                                       transform=transforms.Compose([
                                           transforms.Resize(IMAGE_SIZE),
                                           transforms.PILToTensor(),
                                           transforms.Resize(
                                               IMAGE_SIZE,
                                               antialias=True,
                                               interpolation=InterpolationMode.NEAREST,
                                           ),
                                           transforms.ConvertImageDtype(torch.float32)
                                       ]))

visa_train_dataset = VisaDataset(config.visa_root_path,
                                 config.visa_csv_path,
                                 Split.TRAIN, VisaDatasetCategory.pcb2.value,
                                 gt_mask_size=IMAGE_SIZE,
                                 transform=transforms.Compose([
                                     transforms.Resize(IMAGE_SIZE),
                                     transforms.PILToTensor(),
                                     transforms.Resize(
                                         IMAGE_SIZE,
                                         antialias=True,
                                         interpolation=InterpolationMode.NEAREST,
                                     ),
                                     transforms.ConvertImageDtype(torch.float32)
                                 ]))

visa_test_dataset = VisaDataset(config.visa_root_path,
                                config.visa_csv_path,
                                Split.TEST, VisaDatasetCategory.pcb2.value,
                                gt_mask_size=IMAGE_SIZE,
                                transform=transforms.Compose([
                                    transforms.Resize(IMAGE_SIZE),
                                    transforms.PILToTensor(),
                                    transforms.Resize(
                                        IMAGE_SIZE,
                                        antialias=True,
                                        interpolation=InterpolationMode.NEAREST,
                                    ),
                                    transforms.ConvertImageDtype(torch.float32)
                                ]))
