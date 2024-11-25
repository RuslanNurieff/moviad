import unittest

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import TaskType, Split
from tests.main.common import MVTECH_DATASET_PATH


class MvTecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.training_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            MVTECH_DATASET_PATH,
            'bottle',
            Split.TRAIN,
            img_size=(256, 256),
        )
        self.training_dataset.load_dataset()
        self.test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            MVTECH_DATASET_PATH,
            'bottle',
            Split.TEST,
            img_size=(256, 256),
        )
        self.test_dataset.load_dataset()

    def test_training_dataset_is_contaminated(self):
        initial_training_len = self.training_dataset.__len__()
        initial_test_len = self.test_dataset.__len__()
        self.training_dataset.contaminate(self.test_dataset, 0.1)
        self.assertGreater(self.training_dataset.__len__(), initial_training_len)
        self.assertLess(self.test_dataset.__len__(), initial_test_len)



if __name__ == '__main__':
    unittest.main()
