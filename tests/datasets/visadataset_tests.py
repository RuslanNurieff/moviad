# Description: Real-IAD dataset tests
# This file contains unit tests for the Real-IAD dataset, ensuring proper loading, serialization, and data indexing.
import unittest

import torch
from torchvision.transforms import transforms
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClass
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.utilities.configurations import TaskType, Split
from tests.main.patchcore_tests import transform

VISA_DATASET_PATH = 'E:/VisualAnomalyDetection/datasets/visa'
VISA_DATASET_CSV_PATH = 'E:/VisualAnomalyDetection/datasets/visa/split_csv/1cls.csv'
IMAGE_SIZE = (224, 224)


class VisaTrainDatasetTests(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        self.dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TRAIN, VisaDatasetCategory.candle,
                                   transform=self.transform)

    def test_dataset_is_not_none(self):
        self.assertIsNotNone(self.dataset)

    def test_dataset_should_return_dataset_length(self):
        self.assertIsNotNone(self.dataset.__len__())

    def test_dataset_should_serialize_dataframe(self):
        self.assertIsNotNone(self.dataset.dataframe)
        self.assertEqual(len(self.dataset.dataframe), self.dataset.__len__())

    def test_dataset_should_index_images_and_labels(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.images)
        self.assertIsNotNone(self.dataset.data)
        self.assertEqual(len(self.dataset.data), len(self.dataset.data.images))

    def test_dataset_should_get_item(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.images)
        image = self.dataset.__getitem__(0)
        self.assertIsNotNone(image)
        self.assertIs(type(image), torch.Tensor)
        self.assertEqual(image.shape, torch.Size([3, IMAGE_SIZE[0], IMAGE_SIZE[1]]))


class VisaTestDatasetTests(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        self.dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.candle,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)


    def test_dataset_is_not_none(self):
        self.assertIsNotNone(self.dataset)

    def test_dataset_should_return_dataset_length(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)
        self.assertIsNotNone(self.dataset.__len__())

    def test_dataset_should_serialize_json(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)

    def test_dataset_should_index_images_and_labels(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)
        self.assertIsNotNone(self.dataset.data)
        self.assertEqual(len(self.dataset.data), len(self.dataset.data.data))

    def test_dataset_should_get_item(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.images)
        image, label, mask, path = self.dataset.__getitem__(0)
        self.assertIsNotNone(image)
        self.assertIs(type(image), torch.Tensor)
        self.assertEqual(image.dtype, torch.float32)
        self.assertIsNotNone(label)
        self.assertIs(type(label), int)
        self.assertIn(label, [0, 1])  # 0: normal, 1: abnormal
        self.assertIsNotNone(mask)
        self.assertIs(type(mask), torch.Tensor)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertIsNotNone(path)
        self.assertIs(type(path), str)


if __name__ == '__main__':
    unittest.main()
