import unittest
from dataclasses import dataclass
from tkinter import Image
from tkinter.tix import IMAGE
from typing import List

import torch
from torch.nn.init import normal_
from torchvision.transforms import transforms
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum
from moviad.entrypoints.patchcore import PatchCoreArgs, train_patchcore
from moviad.models.patchcore.patchcore import PatchCore
from moviad.trainers.trainer_patchcore import TrainerPatchCore
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.custom_feature_extractor_trimmed import CustomFeatureExtractor
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
        self.args = PatchCoreArgs()
        self.args.contamination_ratio = 0.25
        self.args.batch_size = 1
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.img_input_size = (224, 224)
        self.args.train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            MVTECH_DATASET_PATH,
            'pill',
            Split.TRAIN,
            img_size=self.args.img_input_size,
        )
        self.args.test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            MVTECH_DATASET_PATH,
            'pill',
            Split.TEST,
            img_size=self.args.img_input_size,
        )
        self.args.train_dataset.load_dataset()
        self.args.test_dataset.load_dataset()
        self.args.category = self.args.train_dataset.category
        self.contamination = 0
        self.args.backbone = BACKBONE
        self.args.ad_layers = AD_LAYERS

    def test_patchcore_with_quantization(self):
        feature_extractor = CustomFeatureExtractor(self.args.backbone, self.args.ad_layers, self.args.device, True,
                                                    False, None)
        train_dataloader = torch.utils.data.DataLoader(self.args.train_dataset, batch_size=self.args.batch_size,
                                                       shuffle=True,
                                                       drop_last=True)
        test_dataloader = torch.utils.data.DataLoader(self.args.test_dataset, batch_size=self.args.batch_size,
                                                      shuffle=True,
                                                      drop_last=True)
        # patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
        #                             feature_extractor=feature_extractor)
        #
        # trainer = TrainerPatchCore(patchcore_model, train_dataloader, test_dataloader, self.args.device,
        #                            apply_quantization=False)
        # trainer.train()
        #
        # normal_memory_bank = patchcore_model.memory_bank
        # self.assertGreater(normal_memory_bank.shape[0], 0)

        patchcore_model = PatchCore(self.args.device, input_size=self.args.img_input_size,
                                    feature_extractor=feature_extractor, apply_quantization=True)
        trainer = TrainerPatchCore(patchcore_model, train_dataloader, test_dataloader, self.args.device,
                                   apply_quantization=True)
        trainer.train()

        quantized_memory_bank = patchcore_model.memory_bank

    def test_patchcore_train_with_quantization(self):
        train_patchcore(self.args)


if __name__ == '__main__':
    unittest.main()
