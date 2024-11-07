import unittest

import torch
from torchvision.models import MobileNet_V2_Weights
from torchvision.transforms import transforms

from main_scripts.main_cfa import train_cfa, train_cfa_v2, test_cfa_v2
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_tests import IMAGE_SIZE, REAL_IAD_DATASET_PATH, AUDIO_JACK_DATASET_JSON
from tests.main.common import get_training_args, MVTECH_DATASET_PATH, REALIAD_DATASET_PATH


class CfaTrainTests(unittest.TestCase):
    def setUp(self):
        self.args = get_training_args()

    def test_cfa_train_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
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

        train_cfa_v2(train_dataset, test_dataset, self.args.category, self.args.backbone,
                     self.args.ad_layers,
                     self.args.epochs,
                     self.args.save_path, self.args.device)

    def test_cfa_train_with_realiad_dataset(self):
        self.args.dataset_path = REALIAD_DATASET_PATH
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        # define training and test datasets
        train_dataset = RealIadDataset(RealIadClass.AUDIOJACK,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=transform)

        test_dataset = RealIadDataset(RealIadClass.AUDIOJACK,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=IMAGE_SIZE,
                                      gt_mask_size=IMAGE_SIZE,
                                      transform=transform)

        train_cfa_v2(train_dataset, test_dataset, 'audiojack', self.args.backbone,
                     self.args.ad_layers,
                     self.args.epochs,
                     self.args.save_path, self.args.device)


class CfaInferenceTests(unittest.TestCase):
    def setUp(self):
        self.args = get_training_args()

    def test_cfa_inference_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
        )

        test_cfa_v2(test_dataset, self.args.category, self.args.backbone,
                    self.args.ad_layers,
                    self.args.model_checkpoint_path,
                    self.args.device)

    def test_cfa_inference_with_realiad_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        test_dataset = RealIadDataset(RealIadClass.AUDIOJACK,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=IMAGE_SIZE,
                                      gt_mask_size=IMAGE_SIZE,
                                      transform=transform)

        test_cfa_v2(test_dataset, 'audiojack', self.args.backbone,
                    self.args.ad_layers,
                    self.args.model_checkpoint_path, self.args.device)


if __name__ == '__main__':
    unittest.main()
