from cProfile import label
from enum import Enum
from os.path import split

import torch
from PIL import Image
import json
from typing import List, Optional
from dataclasses import dataclass
import os
import unittest
from enum import Enum

from numpy.ma.core import masked
from torchvision.transforms import transforms
from pandas import DataFrame

from torch.utils.data import Dataset
from pathlib import Path

from moviad.utilities.configurations import TaskType, Split


class RealIadDefectType(Enum):
    AK = "pit"
    BX = "deformation"
    CH = "abrasion"
    HS = "scratch"
    PS = "damage"
    QS = "missing parts"
    YW = "foreign objects"
    ZW = "contamination"


class RealIadAnomalyClass(Enum):
    OK = "OK"
    NG = "NG"
    AK = "AK"
    BX = "BX"
    CH = "CH"
    HS = "HS"
    PS = "PS"
    QS = "QS"
    YW = "YW"
    ZW = "ZW"

anomaly_class_encoding = {
    RealIadAnomalyClass.OK: 0,
    RealIadAnomalyClass.NG: 1,
    RealIadAnomalyClass.AK: 2,
    RealIadAnomalyClass.BX: 3,
    RealIadAnomalyClass.CH: 4,
    RealIadAnomalyClass.HS: 5,
    RealIadAnomalyClass.PS: 6,
    RealIadAnomalyClass.QS: 7,
    RealIadAnomalyClass.YW: 8,
    RealIadAnomalyClass.ZW: 9,
}


class RealIadClass(Enum):
    AUDIOJACK = "audiojack"
    BOTTLE = "bottle"
    CABLE = "cable"
    CAPSULE = "capsule"
    CARPET = "carpet"
    GRID = "grid"
    HAZELNUT = "hazelnut"
    LEATHER = "leather"
    METAL_NUT = "metal_nut"
    PILL = "pill"
    SCREW = "screw"
    TILE = "tile"
    TOOTHBRUSH = "toothbrush"
    TRANSISTOR = "transistor"
    WOOD = "wood"
    ZIPPER = "zipper"


@dataclass
class MetaData:
    prefix: str
    normal_class: str
    pre_transform: bool


@dataclass
class ImageData:
    category: RealIadClass
    anomaly_class: RealIadAnomalyClass
    image_path: str
    mask_path: Optional[str]  # mask_path may be None

@dataclass
class DatasetImageEntry:
    image: Image
    mask: Image

@dataclass
class RealIadData:
    meta: MetaData
    data: List[ImageData]
    images: List[DatasetImageEntry] = None
    class_name: RealIadClass = None

    @staticmethod
    def from_json(json_path: str, class_name: RealIadClass, split: Split) -> 'RealIadData':
        with open(json_path, 'r') as f:
            data = json.load(f)

        meta = MetaData(**data["meta"])
        train_data = [
            ImageData(category=RealIadClass(item["category"]), anomaly_class=RealIadAnomalyClass(item["anomaly_class"]),
                      image_path=item["image_path"], mask_path=item.get("mask_path")) for item in data[split.value]]
        return RealIadData(meta=meta,
                           data=train_data,
                           class_name=class_name)

    def load_images(self, img_root_dir: str) -> None:
        self.images = []
        image = None
        mask = None
        images_not_found = []
        masks_not_found = []

        for image_entry in self.data:
            class_image_root_path = os.path.join(img_root_dir, self.class_name.value)
            img_path = Path(os.path.join(class_image_root_path, image_entry.image_path))
            if not os.path.exists(img_path):
                images_not_found.append(img_path)
                continue
            image = Image.open(img_path).convert("RGB")

            if image_entry.mask_path is not None:
                image_mask_path = Path(os.path.join(class_image_root_path, image_entry.mask_path))
                if not os.path.exists(image_mask_path):
                    masks_not_found.append(image_mask_path)
                    continue
                mask = Image.open(image_mask_path).convert("L")

            self.images.append(DatasetImageEntry(image=image, mask=mask))


        if len(images_not_found) == self.data.__len__():
            raise ValueError("No images found in the dataset. Check root directory or image paths.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item) -> (ImageData, DatasetImageEntry):
        return self.data[item], self.images[item]


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

    def load_dataset(self) -> None:
        self.data = RealIadData.from_json(self.json, self.class_name, self.split)

        if self.data is None:
            raise ValueError("Dataset is None")

        self.__index_images_and_labels__()

    def __len__(self) -> int:
        return self.data.data.__len__()

    def __getitem__(self, item):
        image_data, image_entry = self.data.__getitem__(item)

        if self.split == Split.TRAIN:
            if self.transform:
                image_entry = self.transform(image_entry.image)
            return image_entry

        if self.split == Split.TEST:
            image = self.transform(image_entry.image)
            label = anomaly_class_encoding[image_data.anomaly_class]
            path = image_data.image_path
            if image_entry.mask is not None:
                mask = image_entry.mask
                mask = self.transform(mask)
            else:
                mask = torch.zeros(1, *self.gt_mask_size)

            return image, label, mask, path

    def __index_images_and_labels__(self) -> None:
        self.data.load_images(self.img_root_dir)
