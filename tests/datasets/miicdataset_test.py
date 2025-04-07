import unittest

from moviad.datasets.builder import DatasetConfig
from moviad.datasets.miic.miic_dataset import MiicDataset
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import TaskType, Split, LabelName


class MiicDatasetTests(unittest.TestCase):
    def setUp(self):
        self.config = DatasetConfig('../config.json')
        self.training_dataset = MiicDataset(
            self.config.miic_train_root_path,
            TaskType.SEGMENTATION,
            Split.TRAIN,
            img_size=(256, 256),
        )
        self.training_dataset.load_dataset()
        self.test_dataset = MiicDataset(
            self.config.miic_test_root_path,
            TaskType.SEGMENTATION,
            Split.TEST,
            img_size=(256, 256),
        )
        # self.test_dataset.load_dataset()

    def test_dataset_is_initialized(self):
        self.assertIsNotNone(self.training_dataset)
        self.assertIsNotNone(self.test_dataset)
        self.assertIsInstance(self.training_dataset, MiicDataset)
        self.assertIsInstance(self.test_dataset, MiicDataset)
        self.assertEqual(self.training_dataset.task, TaskType.SEGMENTATION)
        self.assertEqual(self.training_dataset.split, Split.TRAIN)
        self.assertEqual(self.test_dataset.task, TaskType.SEGMENTATION)
        self.assertEqual(self.test_dataset.split, Split.TEST)

    def test_dataset_should_load_train_images(self):
        self.assertIsNotNone(self.training_dataset.images)
        self.assertGreater(len(self.training_dataset.images), 0)
        self.assertEqual(self.training_dataset.__len__(), len(self.training_dataset.images))




if __name__ == '__main__':
    unittest.main()
