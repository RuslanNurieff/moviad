import unittest
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
from moviad.entrypoints.stfpm import train_stfpm, test_stfpm, visualize_stfpm, STFPMArgs
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_tests import IMAGE_SIZE
from tests.datasets.visadataset_tests import VISA_DATASET_PATH, VISA_DATASET_CSV_PATH


class StfpmTrainTests(unittest.TestCase):
    def setUp(self):
        self.args = STFPMArgs()
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

    def test_Stfpm_train_with_mvtec_dataset(self):
        self.args.dataset_path = self.config.mvtec_root_path
        category = 'pill'
        self.args.categories = [category]
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            category,
            Split.TRAIN,
            img_size=(256, 256),
        )
        self.args.train_dataset = train_dataset

        train_stfpm(train_dataset, self.args)

    def test_Stfpm_train_with_realiad_dataset(self):
        self.categories = [RealIadClassEnum.AUDIOJACK]
        # define training and test datasets
        train_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK.value,
                                       self.config.realiad_root_path,
                                       self.config.realiad_json_root_path,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        self.args.train_dataset = train_dataset
        train_stfpm(self.args)



    def test_Stfpm_train_with_visa_dataset(self):
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.categories = [VisaDatasetCategory.candle.value]
        # define training and test datasets
        train_dataset = VisaDataset(self.config.visa_root_path,
                                    self.config.visa_csv_path,
                                   Split.TRAIN,
                                    VisaDatasetCategory.candle.value,
                                   transform=self.transform)

        self.args.train_dataset = train_dataset
        train_stfpm(self.args)


if __name__ == '__main__':
    unittest.main()