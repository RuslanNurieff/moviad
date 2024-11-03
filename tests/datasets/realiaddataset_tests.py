# Description: Real-IAD dataset tests
# This file contains unit tests for the Real-IAD dataset, ensuring proper loading, serialization, and data indexing.
import unittest

import torch
from torch import tensor
from PIL.Image import Image
from torchvision.transforms import transforms
from moviad.datasets.realiad_dataset import RealIadDataset, RealIadClass
from moviad.utilities.configurations import TaskType, Split

REAL_IAD_DATASET_PATH = 'E:\\VisualAnomalyDetection\\datasets\\Real-IAD\\realiad_256'
AUDIO_JACK_DATASET_JSON = 'E:/VisualAnomalyDetection/datasets/Real-IAD/realiad_jsons/audiojack.json'
IMAGE_SIZE = (224, 224)


class RealIadTrainDatasetTests(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor()
        ])

        self.dataset = RealIadDataset(RealIadClass.AUDIOJACK,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TRAIN,
                                      image_size=IMAGE_SIZE,
                                      transform=self.transform)
        self.dataset.load_dataset()

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
        self.assertIsNotNone(self.dataset.data.data)
        image = self.dataset.__getitem__(0)
        self.assertIsNotNone(image)
        self.assertIs(type(image), torch.Tensor)
        self.assertEqual(image.shape, torch.Size([3, IMAGE_SIZE[0], IMAGE_SIZE[1]]))

    def test_dataset_should_get_item_with_mask(self):
        self.assertIsNotNone(self.dataset.data)
        self.assertIsNotNone(self.dataset.data.meta)
        self.assertIsNotNone(self.dataset.data.data)
        data, image, mask = self.dataset.__getitem__(0)
        self.assertIsNotNone(data)
        self.assertIsNotNone(image)
        self.assertIsNotNone(mask)

class RealIadTestDatasetTests(unittest.TestCase):
    def setUp(self):
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor()
        ])

        self.dataset = RealIadDataset(RealIadClass.AUDIOJACK,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=IMAGE_SIZE,
                                      gt_mask_size=IMAGE_SIZE,
                                      transform=self.transform)
        self.dataset.load_dataset()

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
        self.assertIsNotNone(self.dataset.data.data)
        image, label, mask, path = self.dataset.__getitem__(0)
        self.assertIsNotNone(image)
        self.assertIs(type(image), torch.Tensor)
        self.assertIsNotNone(label)
        self.assertIs(type(label), int)
        self.assertIsNotNone(mask)
        self.assertIs(type(mask), torch.Tensor)
        self.assertIsNotNone(path)
        self.assertIs(type(path), str)



if __name__ == '__main__':
    unittest.main()
