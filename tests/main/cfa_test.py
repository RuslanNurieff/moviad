import unittest

import torch
from torchvision.models import MobileNet_V2_Weights
from torchvision.transforms import transforms, InterpolationMode

from benchmark_config import DatasetConfig
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.entrypoints.cfa import CFAArguments, train_cfa
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_test import IMAGE_SIZE
from tests.datasets.visadataset_test import VISA_DATASET_PATH, VISA_DATASET_CSV_PATH


class CfaTrainTests(unittest.TestCase):
    def setUp(self):
        self.args = CFAArguments()
        self.config = DatasetConfig("./config.yaml")
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.Resize(
                IMAGE_SIZE,
                antialias=True,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ConvertImageDtype(torch.float32)
        ])

    def test_cfa_train_with_mvtec_dataset(self):
        self.args.category = 'pill'
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            self.args.category,
            Split.TRAIN,
            img_size=(256, 256),
        )

        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.config.mvtec_root_path,
            self.args.category,
            Split.TEST,
            img_size=(256, 256),
        )

        train_dataset.load_dataset()
        test_dataset.load_dataset()

        train_cfa(self.args)

    def test_cfa_train_with_realiad_dataset(self):
        # define training and test datasets
        train_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK.value,
                                       self.config.realiad_root_path,
                                       self.config.realiad_json_root_path,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK.value,
                                      self.config.realiad_root_path,
                                      self.config.realiad_json_root_path,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=IMAGE_SIZE,
                                      gt_mask_size=IMAGE_SIZE,
                                      transform=self.transform)

        train_cfa(self.args)

    def test_cfa_train_with_visa_dataset(self):
        self.args.dataset_path = VISA_DATASET_PATH


        # define training and test datasets
        train_dataset = VisaDataset(self.config.visa_root_path,
                                    self.config.visa_csv_path,
                                   Split.TRAIN,
                                    VisaDatasetCategory.candle.value,
                                   transform=self.transform)

        test_dataset = VisaDataset(self.config.visa_root_path,
                                   self.config.visa_csv_path,
                                   Split.TEST, VisaDatasetCategory.candle.value,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)

        train_cfa(self.args)


if __name__ == '__main__':
    unittest.main()
