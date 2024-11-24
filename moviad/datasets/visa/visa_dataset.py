from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from moviad.datasets.common import IadDataset
from moviad.datasets.visa.visa_data import VisaData, VisaAnomalyClass
from moviad.datasets.visa.visa_dataset_configurations import VisaDatasetCategory
from moviad.utilities.configurations import Split, LabelName


class VisaDataset(IadDataset):
    root_path: str
    csv_path: str
    split: Split
    class_name: VisaDatasetCategory
    data: VisaData
    transform: None

    def __init__(self, root_path: str, csv_path: str, split: Split, class_name: VisaDatasetCategory,
                 gt_mask_size: Optional[tuple] = None, transform=None):
        self.root_path = root_path
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        self.class_name = class_name
        self.gt_mask_size = gt_mask_size
        self.dataframe = pd.read_csv(csv_path)
        self.dataframe = self.dataframe[self.dataframe["split"] == split.value]
        self.dataframe = self.dataframe[self.dataframe["object"] == class_name.value]
        self.category = class_name.value

    def set_category(self, category: str):
        self.category = category

    def load_dataset(self):
        self.__load__()

    def contaminate(self, source: 'IadDataset', ratio: float, seed: int = 42) -> None:
        if not isinstance(source, VisaDataset):
            raise ValueError("Dataset should be of type VisaDataset")
        if self.data is None or self.data.data is None:
            raise ValueError("Destination dataset is not loaded")
        if source.data is None or source.data.data is None:
            raise ValueError("Source dataset is not loaded")

        torch.manual_seed(seed)
        contamination_set_size = int(len(self.data) * ratio)
        while contamination_set_size > 0:
            index = torch.randint(0, len(source.data.images), (1,)).item()
            entry = source.data.images[index]
            if entry.label == VisaAnomalyClass.NORMAL:
                continue
            if self.data.images.__contains__(entry):
                continue
            self.data.images.append(entry)
            source.data.images.remove(entry)
            contamination_set_size -= 1


    def __load__(self):
        self.data = VisaData(meta=self.dataframe, data=self.dataframe)
        self.data.load_images(self.root_path, split=self.split)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        image_data_entry = self.data.images[item]
        image = image_data_entry.image
        mask = image_data_entry.mask

        if self.split == Split.TRAIN:
            if self.transform:
                image = self.transform(image)
            return image

        if self.split == Split.TEST:
            label = LabelName.NORMAL.value if image_data_entry.label == VisaAnomalyClass.NORMAL else LabelName.ABNORMAL.value
            path = str(image_data_entry.image_path)
            if mask is not None:
                mask = self.transform(mask)
            else:
                mask = torch.zeros(1, *self.gt_mask_size, dtype=torch.float32)
            if self.transform:
                image = self.transform(image)

            return image, label, mask, path
