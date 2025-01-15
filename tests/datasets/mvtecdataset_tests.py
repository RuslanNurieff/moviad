import unittest

from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.utilities.configurations import TaskType, Split, LabelName
from tests.main.common import MVTECH_DATASET_PATH


class MvTecDatasetTests(unittest.TestCase):
    def setUp(self):
        self.training_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            MVTECH_DATASET_PATH,
            'pill',
            Split.TRAIN,
            img_size=(256, 256),
        )
        self.training_dataset.load_dataset()
        self.test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            MVTECH_DATASET_PATH,
            'pill',
            Split.TEST,
            img_size=(256, 256),
        )
        self.test_dataset.load_dataset()

    def test_training_dataset_is_contaminated(self):
        initial_training_len = self.training_dataset.__len__()
        initial_test_len = self.test_dataset.__len__()
        contamination_size = self.training_dataset.contaminate(self.test_dataset, 0.1)

        contaminated_entries = self.training_dataset.samples[
            self.training_dataset.samples["label_index"] == LabelName.ABNORMAL.value]

        self.assertGreater(self.training_dataset.__len__(), initial_training_len)
        self.assertLess(self.test_dataset.__len__(), initial_test_len)
        self.assertEqual(contamination_size, abs(initial_training_len - self.training_dataset.__len__()))
        self.assertEqual(contamination_size, abs(initial_test_len - self.test_dataset.__len__()))
        self.assertEqual(contamination_size, len(contaminated_entries))
        contamintation_ratio = self.training_dataset.compute_contamination_ratio()
        self.assertLess(contamintation_ratio, 1.0)
        self.assertGreater(contamintation_ratio, 0.0)


if __name__ == '__main__':
    unittest.main()
