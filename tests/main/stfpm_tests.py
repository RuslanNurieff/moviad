import unittest
import unittest

import torch
from torchvision.models import MobileNet_V2_Weights
from torchvision.transforms import transforms, InterpolationMode
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.entrypoints.stfpm import train_stfpm, test_stfpm, visualize_stfpm
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_tests import IMAGE_SIZE, REAL_IAD_DATASET_PATH, AUDIO_JACK_DATASET_JSON
from tests.datasets.visadataset_tests import VISA_DATASET_PATH, VISA_DATASET_CSV_PATH
from tests.main.common import get_training_args, MVTECH_DATASET_PATH, REALIAD_DATASET_PATH, get_stfpm_train_args, \
    get_stfpm_test_args


class StfpmTrainTests(unittest.TestCase):
    def setUp(self):
        self.args = get_stfpm_train_args()

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
        self.args.dataset_path = MVTECH_DATASET_PATH
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
        self.args.dataset_path = REALIAD_DATASET_PATH
        self.categories = [RealIadClassEnum.AUDIOJACK]
        # define training and test datasets
        train_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
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
        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TRAIN,
                                    VisaDatasetCategory.candle,
                                   transform=self.transform)

        self.args.train_dataset = train_dataset
        train_stfpm(self.args)


class StfpmInferenceTests(unittest.TestCase):
    def setUp(self):
        self.args = get_stfpm_test_args()
        self.args.model_checkpoint_path = '.'
        self.args.model_name = 'mobilenet_v2'
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

    def test_cfa_inference_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        self.args.categories = ['pill']
        self.args.class_name = 'pill'
        self.args.feature_maps_dir = '.'
        self.args.input_size = (224, 224)
        self.args.output_size = (224, 224)
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.class_name,
            Split.TEST,
            img_size=(224, 224),
        )
        test_dataset.load_dataset()
        self.args.test_dataset = test_dataset
        test_stfpm(self.args)
        visualize_stfpm(self.args)

    def test_cfa_inference_with_realiad_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])
        self.args.dataset_path = REALIAD_DATASET_PATH
        self.args.class_name = RealIadClassEnum.AUDIOJACK
        self.args.categories = [RealIadClassEnum.AUDIOJACK]
        self.args.feature_maps_dir = '.'
        self.args.input_size = (224, 224)
        self.args.output_size = (224, 224)
        test_dataset = RealIadDataset(self.args.class_name,
                                      self.args.dataset_path,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=self.args.input_size,
                                      gt_mask_size=self.args.output_size,
                                      transform=transform)
        test_dataset.load_dataset()
        self.args.test_dataset = test_dataset
        test_stfpm(self.args)
        visualize_stfpm(self.args)

    def test_Stfpm_inference_with_visa_dataset(self):
        self.args.dataset_path = VISA_DATASET_PATH
        self.args.class_name = 'candle'
        self.args.input_size = (224, 224)
        self.args.output_size = (224, 224)

        test_dataset = VisaDataset(self.args.dataset_path,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST,
                                   VisaDatasetCategory.candle,
                                   gt_mask_size=self.args.input_size,
                                   transform=self.transform)
        test_dataset.load_dataset()
        self.args.test_dataset = test_dataset

        test_stfpm(self.args)


if __name__ == '__main__':
    unittest.main()