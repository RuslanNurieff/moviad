import unittest
import unittest

import torch
from torchvision.models import MobileNet_V2_Weights
from torchvision.transforms import transforms, InterpolationMode

from main_scripts.main_cfa import main_train_cfa, train_cfa_v2, test_cfa_v2
from main_scripts.main_padim import main_train_padim, test_padim
from moviad.datasets.mvtec.mvtec_dataset import MVTecDataset
from moviad.datasets.realiad.realiad_dataset import RealIadDataset
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClassEnum
from moviad.datasets.visa.visa_dataset import VisaDataset
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.utilities.configurations import TaskType, Split
from tests.datasets.realiaddataset_tests import IMAGE_SIZE, REAL_IAD_DATASET_PATH, AUDIO_JACK_DATASET_JSON
from tests.datasets.visadataset_tests import VISA_DATASET_PATH, VISA_DATASET_CSV_PATH
from tests.main.common import get_training_args, MVTECH_DATASET_PATH, REALIAD_DATASET_PATH



class PadimTrainTests(unittest.TestCase):
    def setUp(self):
        self.args = get_training_args()
        self.args.model_checkpoint_path = '.'
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.Resize(
                IMAGE_SIZE,
                antialias=True,
                interpolation=InterpolationMode.NEAREST,
            ),
            transforms.ConvertImageDtype(torch.float32),

        ])

    def test_padim_train_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        train_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.class_name,
            Split.TRAIN,
            img_size=(256, 256),
        )


        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.class_name,
            Split.TEST,
            img_size=(256, 256),
        )


        main_train_padim(train_dataset, test_dataset, self.args.class_name, self.args.backbone,
                         self.args.ad_layers, self.args.device, self.args.model_checkpoint_path)

    def test_padim_train_with_realiad_dataset(self):
        self.args.dataset_path = REALIAD_DATASET_PATH

        # define training and test datasets
        train_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK,
                                       REAL_IAD_DATASET_PATH,
                                       AUDIO_JACK_DATASET_JSON,
                                       task=TaskType.SEGMENTATION,
                                       split=Split.TRAIN,
                                       image_size=IMAGE_SIZE,
                                       transform=self.transform)

        test_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=IMAGE_SIZE,
                                      gt_mask_size=IMAGE_SIZE,
                                      transform=self.transform)

        main_train_padim(train_dataset, test_dataset, 'audiojack', self.args.backbone,
                         self.args.ad_layers,
                         self.args.device, self.args.model_checkpoint_path)



    def test_padim_train_with_visa_dataset(self):
        self.args.dataset_path = VISA_DATASET_PATH


        # define training and test datasets
        train_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TRAIN,
                                    VisaDatasetCategory.candle,
                                   transform=self.transform)

        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.candle,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)

        main_train_padim(train_dataset, test_dataset, 'candle', self.args.backbone,
                         self.args.ad_layers,
                         self.args.device, self.args.model_checkpoint_path)


class PadimInferenceTests(unittest.TestCase):
    def setUp(self):
        self.args = get_training_args()
        self.args.model_checkpoint_path = '.'
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

    def test_cfa_inference_with_mvtec_dataset(self):
        self.args.dataset_path = MVTECH_DATASET_PATH
        test_dataset = MVTecDataset(
            TaskType.SEGMENTATION,
            self.args.dataset_path,
            self.args.class_name,
            Split.TEST,
            img_size=(256, 256),
        )

        test_padim(test_dataset, self.args.class_name, self.args.backbone,
                   self.args.ad_layers,
                   self.args.device, self.args.model_checkpoint_path)

    def test_cfa_inference_with_realiad_dataset(self):
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ])

        test_dataset = RealIadDataset(RealIadClassEnum.AUDIOJACK,
                                      REAL_IAD_DATASET_PATH,
                                      AUDIO_JACK_DATASET_JSON,
                                      task=TaskType.SEGMENTATION,
                                      split=Split.TEST,
                                      image_size=IMAGE_SIZE,
                                      gt_mask_size=IMAGE_SIZE,
                                      transform=transform)

        test_padim(test_dataset, 'audiojack', self.args.backbone,
                    self.args.ad_layers,
                    self.args.device, self.args.model_checkpoint_path)

    def test_padim_inference_with_visa_dataset(self):
        self.args.dataset_path = VISA_DATASET_PATH


        test_dataset = VisaDataset(VISA_DATASET_PATH,
                                   VISA_DATASET_CSV_PATH,
                                   Split.TEST, VisaDatasetCategory.candle,
                                   gt_mask_size=IMAGE_SIZE,
                                   transform=self.transform)

        test_padim(test_dataset, 'candle', self.args.backbone,
                    self.args.ad_layers,
                    self.args.device,  self.args.model_checkpoint_path)


if __name__ == '__main__':
    unittest.main()