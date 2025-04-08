import unittest

import torch

from moviad.datasets.builder import DatasetConfig
from moviad.datasets.miic.miic_dataset import MiicDataset, MiicDatasetConfig
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import TaskType, Split, LabelName


class MiicDatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig('../config.json')
        self.training_dataset = MiicDataset(MiicDatasetConfig(
            training_root_path=self.config.miic_train_root_path,
            split=Split.TRAIN,
            task_type=TaskType.CLASSIFICATION))
        self.training_dataset.load_dataset()
        self.test_dataset = MiicDataset(MiicDatasetConfig(
            test_abnormal_image_root_path=self.config.miic_test_abnormal_image_root_path,
            test_normal_image_root_path=self.config.miic_test_normal_image_root_path,
            test_abnormal_bounding_box_root_path=self.config.miic_test_abnormal_bounding_box_root_path,
            test_abnormal_mask_root_path=self.config.miic_test_abnormal_mask_root_path,
            split=Split.TEST,
            task_type=TaskType.CLASSIFICATION)
        )
        self.test_dataset.load_dataset()

    def test_dataset_is_initialized(self):
        self.assertIsNotNone(self.training_dataset)
        self.assertIsNotNone(self.test_dataset)
        self.assertIsInstance(self.training_dataset, MiicDataset)
        self.assertIsInstance(self.test_dataset, MiicDataset)
        self.assertEqual(self.training_dataset.task, TaskType.CLASSIFICATION)
        self.assertEqual(self.training_dataset.split, Split.TRAIN)
        self.assertEqual(self.test_dataset.task, TaskType.CLASSIFICATION)
        self.assertEqual(self.test_dataset.split, Split.TEST)

    def test_dataset_should_load_images(self):
        self.assertIsNotNone(self.training_dataset.images)
        self.assertGreater(len(self.training_dataset.images), 0)
        self.assertEqual(self.training_dataset.__len__(), len(self.training_dataset.images))

        self.assertIsNotNone(self.test_dataset.images)
        self.assertGreater(len(self.test_dataset.images), 0)
        self.assertEqual(self.test_dataset.__len__(), len(self.test_dataset.images))
        self.assertIsNotNone(self.test_dataset.normal_images_entries)
        self.assertIsNotNone(self.test_dataset.abnormal_images_entries)
        self.assertEqual(len(self.test_dataset.images), len(self.test_dataset.normal_images_entries) + len(
            self.test_dataset.abnormal_images_entries))

    def test_dataset_should_get_item(self):
        item = self.training_dataset.__getitem__(0)
        self.assertIsNotNone(item)
        self.assertIsInstance(item, torch.Tensor)




if __name__ == '__main__':
    unittest.main()
