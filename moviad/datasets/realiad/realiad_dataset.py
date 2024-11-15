from cProfile import label
from enum import Enum
from os.path import split

import torch
from PIL import Image
import json
from typing import List, Optional
import os

from torch.utils.data import Dataset
from pathlib import Path

from moviad.datasets.realiad.realiad_data import RealIadData
from moviad.datasets.realiad.realiad_dataset_configurations import RealIadClass, RealIadAnomalyClass
from moviad.utilities.configurations import TaskType, Split, LabelName


class RealIadDataset(Dataset):
    def __init__(self, class_name: RealIadClass, img_root_dir: str, json_path: str, task: TaskType, split: Split,
                 gt_mask_size: Optional[tuple] = None,
                 transform=None,
                 image_size=(224, 224)) -> None:
        super().__init__()
        if img_root_dir is None:
            raise ValueError("img_dir should not be None")
        if not os.path.exists(img_root_dir):
            raise ValueError(f"img_dir '{img_root_dir}' does not exist")
        if not os.path.isdir(img_root_dir):
            raise ValueError(f"img_dir '{img_root_dir}' is not a directory")
        self.json = json_path
        self.img_root_dir = img_root_dir
        self.transform = transform
        self.class_name = class_name
        self.data: RealIadData = None
        self.task = task
        self.split = split
        self.gt_mask_size = gt_mask_size
        self.load_dataset()

    def load_dataset(self) -> None:
        self.data = RealIadData.from_json(self.json, self.class_name, self.split)

        if self.data is None:
            raise ValueError("Dataset is None")

        self.__index_images_and_labels__()

    def __len__(self) -> int:
        return self.data.__len__()

    def __getitem__(self, item):
        image_data, image_entry = self.data.__getitem__(item)

        if self.split == Split.TRAIN:
            if self.transform:
                image_entry = self.transform(image_entry.image)
            return image_entry

        if self.split == Split.TEST:
            image = self.transform(image_entry.image)
            label = LabelName.NORMAL.value if image_data.anomaly_class == RealIadAnomalyClass.OK else LabelName.ABNORMAL.value
            path = image_data.image_path
            if image_entry.mask is not None:
                mask = image_entry.mask
                mask = self.transform(mask)
            else:
                mask = torch.zeros(1, *self.gt_mask_size, dtype=torch.float32)

            return image, label, mask, path

    def __index_images_and_labels__(self) -> None:
        self.data.load_images(self.img_root_dir)
