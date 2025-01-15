# Description: Real-IAD dataset tests
# This file contains unit tests for the Real-IAD dataset, ensuring proper loading, serialization, and data indexing.
import unittest

import torch
from PIL import ImageEnhance
from torch.onnx.symbolic_opset9 import tensor
from torchvision.transforms import transforms

from benchmark_config import DatasetConfig
from moviad.datasets.realiad.realiad_dataset import RealIadDataset, RealIadClassEnum
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadAnomalyClass
from moviad.datasets.visa.visa_data import VisaAnomalyClass
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.utilities.configurations import TaskType, Split
from moviad.utilities.pil_image_utils import IncreaseContrast
from tests.main.patchcore_tests import transform
IMAGE_SIZE = (224, 224)


class VisaTrainDatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig("./config.yaml")
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        self.dataset = VisaDataset(self.config.visa_root_path,
                                   self.config.visa_csv_path,
                                   Split.TRAIN,
                                   VisaDatasetCategory.candle.value,
                                   transform=self.transform)

        self.dataset.load_dataset()


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


class VisaDatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig("./config.yaml")
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            IncreaseContrast(1.5),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        self.train_dataset = VisaDataset(self.config.visa_root_path,
                                         self.config.visa_csv_path,
                                         Split.TRAIN, VisaDatasetCategory.pipe_fryum.value,
                                         gt_mask_size=IMAGE_SIZE,
                                         transform=self.transform)

        self.train_dataset.load_dataset()
        self.test_dataset = VisaDataset(self.config.visa_root_path,
                                        self.config.visa_csv_path,
                                        Split.TEST, VisaDatasetCategory.pipe_fryum.value,
                                        gt_mask_size=IMAGE_SIZE,
                                        transform=self.transform)
        self.test_dataset.load_dataset()

    def test_check_all_mask_are_not_none(self):
        masks = [image_entry.mask for image_entry in self.train_dataset.data.images]
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            IncreaseContrast(3.5),
            transforms.PILToTensor(),
        ])
        masks = [mask for mask in masks if mask is not None]
        enhancer = ImageEnhance.Contrast(masks[0])
        masks[0] = enhancer.enhance(1.5)
        masks[0].show()
        mask_tensors = [transform(mask) for mask in masks if mask is not None]
        assert not torch.any(sum(mask_tensors)), "The tensor is all zeros"

    def test_dataset_is_not_none(self):
        self.assertIsNotNone(self.train_dataset)

    def test_dataset_should_return_dataset_length(self):
        self.assertIsNotNone(self.train_dataset.data)
        self.assertIsNotNone(self.train_dataset.data.meta)
        self.assertIsNotNone(self.train_dataset.data.data)
        self.assertIsNotNone(self.train_dataset.__len__())

    def test_dataset_should_serialize_json(self):
        self.assertIsNotNone(self.train_dataset.data)
        self.assertIsNotNone(self.train_dataset.data.meta)
        self.assertIsNotNone(self.train_dataset.data.data)

    def test_dataset_should_index_images_and_labels(self):
        self.assertIsNotNone(self.train_dataset.data)
        self.assertIsNotNone(self.train_dataset.data.meta)
        self.assertIsNotNone(self.train_dataset.data.data)
        self.assertIsNotNone(self.train_dataset.data)
        self.assertEqual(len(self.train_dataset.data), len(self.train_dataset.data.data))

    def test_dataset_should_get_item(self):
        self.assertIsNotNone(self.train_dataset.data)
        self.assertIsNotNone(self.train_dataset.data.meta)
        self.assertIsNotNone(self.train_dataset.data.images)
        image, label, mask, path = self.train_dataset.__getitem__(0)
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

    def test_training_dataset_should_not_contain_anoamlies(self):
        for item in self.train_dataset.data.images:
            self.assertEqual(item.label, VisaAnomalyClass.NORMAL)

    def test_test_dataset_should_contain_anoamlies(self):
        contains_anomalies = False
        for item in self.test_dataset.data.images:
            if item.label != VisaAnomalyClass.NORMAL:
                contains_anomalies = True
                break
        self.assertTrue(contains_anomalies)

    def test_training_dataset_is_contaminated(self):
        initial_train_size = self.train_dataset.__len__()
        initial_test_size = self.test_dataset.__len__()
        contamination_size = self.train_dataset.contaminate(self.test_dataset, 0.1)

        contaminated_entries = [entry for entry in self.train_dataset.data.images if
                                entry.label == VisaAnomalyClass.ANOMALY]

        self.assertGreater(len(contaminated_entries), 0)
        self.assertGreater(self.train_dataset.__len__(), initial_train_size)
        self.assertLess(self.test_dataset.__len__(), initial_test_size)
        self.assertEqual(contamination_size, abs(initial_train_size - self.train_dataset.__len__()))
        self.assertEqual(contamination_size, abs(initial_test_size - self.test_dataset.__len__()))
        self.assertEqual(contamination_size, len(contaminated_entries))

        contamination_ratio = self.train_dataset.compute_contamination_ratio()
        self.assertLess(contamination_ratio, 1.0)
        self.assertGreater(contamination_ratio, 0.0)



if __name__ == '__main__':
    unittest.main()
